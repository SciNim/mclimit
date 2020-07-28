import mersenne, logging, strformat, math

import arraymancer

type
  Histogram = object # can be ndimensional in principle
    ndim: int # dimensionality for reference
    bins: Tensor[float]
    counts: Tensor[float]
    err: Tensor[float]

  # a systematic error for candidate and background channel
  SystematicError = object
    cand: float
    back: float

  DataSource = object
    sig: seq[Histogram] # expected signal hypothesis, one for each channel
    back: seq[Histogram] # measured background
    cand: seq[Histogram] # measured candidates
    systErr: OrderedTable[string, SystematicError]

  ConfidenceLevel = object
    btot: float
    stot: float
    dtot: float
    tsd: float

# set up the logger
var L = newConsoleLogger()
if not dirExists("logs"):
  createDir("logs")
var fL = newFileLogger("logs/mclimit.log", fmtStr = verboseFmtStr)
when isMainModule:
  addHandler(L)
  addHandler(fL)

proc getBins(h: Histogram): int =
  assert h.ndim == 1
  result = h.bins.size

proc logLikelihood(s, b, b2, d: float): float =
  result = d * ln((s + b) / b2)

template rand(rng: MersenneTwister): float =
  # 2.3283064365386963e-10 == 1./(max<UINt_t>+1)  -> then returned value cannot be = 1.0
  rng.getNum().float * 2.3283064365386963e-10 # * Power(2,-32)

proc gauss(rnd: MersenneTwister, mean, sigma: float): float =
  ## based on stdlib, which uses:
  # Ratio of uniforms method for normal
  # http://www2.econ.osaka-u.ac.jp/~tanizaki/class/2013/econome3/13.pdf
  const K = sqrt(2 / E)
  var
    a = 0.0
    b = 0.0
  while true:
    a = rand(rnd)
    b = (2.0 * rand(rnd) - 1.0) * K
    if  b * b <= -4.0 * a * a * ln(a): break
  result = mu + sigma * (b / a)

proc computeLimit(data: DataSource, rnd: MersenneTwister): ConfidenceLevel =
  # determine the number of bins the channel with most bins has
  let nChannel = data.sig.len
  let maxBins = max(data.sig.mapIt(it.getBins + 2))
  let nsig = data.sig.foldl(a.counts.sum + b.counts.sum, 0.0)
  let nbg = data.back.foldl(a.counts.sum + b.counts.sum, 0.0)
  let ncand = data.cand.foldl(a.counts.sum + b.counts.sum, 0.0)

  result = ConfidenceLevel(btot: nbg, stot: nsig, dtot: ncand)

  var fgTable = newTensor[float]((maxbins, nChannel))
  var buffer = 0.0
  for chIdx in 0 ..< nChannel:
    for bin in 0 ..< data.sig[chIdx].getBins:
      let s = data.sig[chIdx][bin]
      let b = data.back[chIdx][bin]
      let d = data.cand[chIdx][bin]
      if almostEqual(b, 0.0) and almostEqual(s, 0.0):
        warn &"Ignoring bin {bin} of channel {channel} which has s = {s} but" &
          &" b = {b}\n\tMaybe the MC statistic has to be improved..."
      if s > 0.0 and b > 0.0:
        buffer += logLikelihood(s, b, b, d)
        fgTable[channel, bin] = logLikelihood(s, b, b, 1)
      elif s > 0.0 and almostEqual(b, 0.0):
        fgTable[channel, bin] = 20.0 # why do we add 20? Yeah, that's a large value for a logL but still

  result.tsd = buffer

  ## now comes the monte carlo part
  var
    tss = zeros[float](nmc)
    tsb = zeros[float](nmc)
    lrs = zeros[float](nmc)
    lrb = zeros[float](nmc)
  for i in 0 ..< nmc:
    discard

proc fluctuate(input: DataSource, output: var DataSource,
               init: bool, rnd: MersenneTwister, stat: bool): bool =
  discard
