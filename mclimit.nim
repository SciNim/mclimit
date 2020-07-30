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

  Channel = object
    sig: Histogram # expected signal hypothesis, one for each channel
    back: Histogram # measured background
    cand: Histogram # measured candidates
    systErr: OrderedTable[string, SystematicError]

  DataSource = seq[Channel]

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
  let nChannel = data.len
  let maxBins = max(data.sig.mapIt(it.getBins + 2))
  template sumIt(fd: untyped): untyped = data.foldl(a.fd.counts.sum + b.fd.counts.sum, 0.0)
  let nsig = sumIt(sig)
  let nbg = sumIt(back)
  let ncand = sumIt(cand)

  result = ConfidenceLevel(btot: nbg, stot: nsig, dtot: ncand)

  var fgTable = newTensor[float]((maxbins, nChannel))
  var buffer = 0.0
  for chIdx in 0 ..< nChannel:
    for bin in 0 ..< data[chIdx].sig.getBins:
      let s = data[chIdx].sig[bin]
      let b = data[chIdx].back[bin]
      let d = data[chIdx].cand[bin]
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
  if output.sig.len == 0: # should imply output isn't initialized yet
    output = input
  if input.systErr.len == 0 and not stat:
    # if there are no systematics and we don't use statistical errors, we cannot
    # fluctuate or in other words the input ``is`` the fluctuated output
    return false
  elif input.systErr.len == 0:
    # in this case just fluctuate using statistics
    for chIdx in 0 ..< nChannel:
      template statFluc(field: untyped): untyped =
        var new = output.field[chIdx]
        var old = input.field[chIdx]
        for bin in 0 ..< data.field[chIdx].getBins:
          new.count[bin] = old.count[bin] + rnd.gauss(0, old.err[bin])
        output.field[chIdx] = new # make ref object to alleviate these copies?
      statFluc(sig)
      statFluc(back)
    return true
  else:
    # else use both statistical and systematic
    discard
