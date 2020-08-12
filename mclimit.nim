import logging, strformat, math, tables, os, sequtils

import alea, arraymancer

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
    tss: Tensor[float]
    tsb: Tensor[float]
    lrs: Tensor[float]
    lrb: Tensor[float]

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

template random(rng: Random, uniform: Uniform): float =
  # 2.3283064365386963e-10 == 1./(max<UINt_t>+1)  -> then returned value cannot be = 1.0
  rng.sample(uniform) * 2.3283064365386963e-10 # * Power(2,-32)

template almostEqual(x, y, eps = 1e-8): untyped =
  # TODO: in principle we want a more rigorous check, but that will be more
  # costly. The question is whether the check needs to be safe or whether numbers
  # will be sane
  abs(x - y) < eps

proc gaus(rnd: var Random, mean, sigma: float): float =
  ## based on stdlib, which uses:
  # Ratio of uniforms method for normal
  # http://www2.econ.osaka-u.ac.jp/~tanizaki/class/2013/econome3/13.pdf
  const K = sqrt(2 / E)
  const uni = uniform(0, 1)
  var
    a = 0.0
    b = 0.0
  while true:
    a = random(rnd, uni)
    b = (2.0 * random(rnd, uni) - 1.0) * K
    if  b * b <= -4.0 * a * a * ln(a): break
  result = mean + sigma * (b / a)

proc fluctuate(input: DataSource, output: var DataSource,
               rnd: var Random, stat: bool): bool =
  template statFluc(chIdx, field: untyped): untyped =
    var new = output[chIdx].field
    var old = input[chIdx].field
    if stat:
      for bin in 0 ..< new.getBins:
        new.counts[bin] = old.counts[bin] + gaus(rnd, 0.0, old.err[bin])
    else:
      new = old
    new

  let nChannel = input.len
  if output.len == 0: # should imply output isn't initialized yet
    output = input
  if input[0].systErr.len == 0 and not stat:
    # if there are no systematics and we don't use statistical errors, we cannot
    # fluctuate or in other words the input ``is`` the fluctuated output
    return false
  elif input[0].systErr.len == 0:
    # in this case just fluctuate using statistics
    for chIdx in 0 ..< nChannel:
      # make ref object to alleviate these copies?
      output[chIdx].sig = statFluc(chIdx, sig)
      output[chIdx].back = statFluc(chIdx, back)
    return true
  else:
    # else use both statistical and systematic
    # Find a choice for the random variation and
    # re-toss all random numbers if any background or signal
    # goes negative.  (background = 0 is bad too, so put a little protection
    # around it -- must have at least 10% of the bg estimate).
    var
      reToss = true
      serrf = zeros[float](input.len)
      berrf = zeros[float](input.len)
    while reToss:
      var toss = zeros[float](input[0].systErr.len)
      toss.apply_inline(rnd.gaus(0.0, 1.0))
      reToss = false
      for chIdx in 0 ..< input.len:
        serrf[chIdx] = 0.0
        berrf[chIdx] = 0.0
        var tIdx = 0
        for key, val in input[chIdx].systErr:
          serrf[chIdx] += val.cand * toss[tIdx]
          berrf[chIdx] += val.back * toss[tIdx]
          inc tIdx
        if serrf[chIdx] < -1.0 or berrf[chIdx] < -0.9:
          reToss = true
          continue
    # now apply statistical error too
    for chIdx in 0 ..< nChannel:
      var newSig = statFluc(chIdx, sig)
      newSig.counts.apply_inline(x * (1.0 + serrf[chIdx]))
      output[chIdx].sig = newSig
      var newBack = statFluc(chIdx, back)
      newBack.counts.apply_inline(x * (1.0 + berrf[chIdx]))
      output[chIdx].back = newBack
    result = true

proc computeLimit(data: DataSource, rnd: var Random,
                  stat: bool,
                  nmc: int = 1_000_000): ConfidenceLevel =
  # determine the number of bins the channel with most bins has
  let nChannel = data.len
  let maxBins = max(data.mapIt(it.sig.getBins + 2))
  template sumIt(fd: untyped): untyped = data.mapIt(it.fd.counts.sum).sum
  let nsig = sumIt(sig)
  let nbg = sumIt(back)
  let ncand = sumIt(cand)

  result = ConfidenceLevel(btot: nbg, stot: nsig, dtot: ncand)

  var fgTable = newTensor[float]([maxbins, nChannel])
  var buffer = 0.0
  for chIdx in 0 ..< nChannel:
    for bin in 0 ..< data[chIdx].sig.getBins:
      let s = data[chIdx].sig.counts[bin]
      let b = data[chIdx].back.counts[bin]
      let d = data[chIdx].cand.counts[bin]
      if almostEqual(b, 0.0) and almostEqual(s, 0.0):
        warn &"Ignoring bin {bin} of channel {chIdx} which has s = {s} but" &
          &" b = {b}\n\tMaybe the MC statistic has to be improved..."
      if s > 0.0 and b > 0.0:
        buffer += logLikelihood(s, b, b, d)
        fgTable[chIdx, bin] = logLikelihood(s, b, b, 1)
      elif s > 0.0 and almostEqual(b, 0.0):
        fgTable[chIdx, bin] = 20.0 # why do we add 20? Yeah, that's a large value for a logL but still

  result.tsd = buffer

  ## now comes the monte carlo part
  var
    tss = zeros[float](nmc)
    tsb = zeros[float](nmc)
    lrs = zeros[float](nmc)
    lrb = zeros[float](nmc)
    tmp1 = data
    tmp2 = data
    pois: Poisson
  for i in 0 ..< nmc:
    let fluct1 = if fluctuate(data, tmp1, rnd, stat): tmp1 else: data
    let fluct2 = if fluctuate(data, tmp2, rnd, stat): tmp2 else: data
    for chIdx in 0 ..< nChannel:
      for bin in 0 ..< fluct1[chIdx].sig.getBins:
        if fluct1[chIdx].sig.counts[bin] != 0.0:
          # s+b hypothesis
          var rate = fluct1[chIdx].sig.counts[bin] + fluct1[chIdx].back.counts[bin]
          pois = poisson(rate)
          var rand = rnd.sample(pois)
          tss[i] += rand * fgTable[chIdx, bin]
          let
            s = fluct1[chIdx].sig.counts[bin]
            s2 = fluct2[chIdx].sig.counts[bin]
            b = fluct1[chIdx].back.counts[bin]
            b2 = fluct2[chIdx].back.counts[bin]
          if s > 0.0 and b2 > 0.0:
            lrs[i] += logLikelihood(s, b, b2, rand) - s - b + b2
          elif s > 0.0 and b2 == 0.0:
            lrs[i] += 20.0 * rand - s
          # b hypothesis
          rate = fluct1[chIdx].back.counts[bin]
          pois = poisson(rate)
          rand = rnd.sample(pois)
          tsb[i] += rand * fgTable[chIdx, bin]
          if s2 > 0.0 and b > 0.0:
            lrb[i] += logLikelihood(s2, b2, b, rand) - s2 - b2 + b
          # TODO: is it correct here is `s` again instead of s2? seems weird
          elif s > 0.0 and b2 == 0.0:
            lrb[i] += 20.0 * rand - s
    lrs[i] = if lrs[i] < 710: exp(lrs[i]) else: exp(710.0)
    lrb[i] = if lrb[i] < 710: exp(lrb[i]) else: exp(710.0)

  result.tss = tss
  result.tsb = tsb
  result.lrs = lrs
  result.lrb = lrb
