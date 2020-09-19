import logging, strformat, math, tables, os, sequtils, sugar

import arraymancer / tensor
export tensor
import random / mersenne
export mersenne
#import alea except `+` # Tensor seems to suit RandomVar concept for some reason,
                        # which calls + for RandomVar in arraymancer
# also cannot import sugar, since that breaks in combination with ggplotnim / chroma
import alea / [core, rng, gauss, poisson]
export core, rng, gauss, poisson
import seqmath
type
  Histogram* = object # can be ndimensional in principle
    ndim*: int # dimensionality for reference
    bins*: Tensor[float]
    counts*: Tensor[float]
    err*: Tensor[float]

  # a systematic error for candidate and background channel
  SystematicError* = object
    cand*: float
    back*: float

  Channel* = object
    sig*: Histogram # expected signal hypothesis, one for each channel
    back*: Histogram # measured background
    cand*: Histogram # measured candidates
    systErr*: OrderedTable[string, SystematicError]

  DataSource* = seq[Channel]

  ConfidenceLevel* = object
    nmc*: int # number of monte carlo samples
    btot*: float
    stot*: float
    dtot*: float
    tsd*: float
    iss*: Tensor[int]
    isb*: Tensor[int]
    tss*: Tensor[float]
    tsb*: Tensor[float]
    lrs*: Tensor[float]
    lrb*: Tensor[float]

const
  fgMCLM2S = 0.025
  fgMCLM1S = 0.16
  fgMCLMED = 0.5
  fgMCLP1S = 0.84
  fgMCLP2S = 0.975
  # one-sided" definition
  fgMCL3S1S = 2.6998E-3
  fgMCL5S1S = 5.7330E-7
  # the other definition (not chosen by the LHWG)
  fgMCL3S2S = 1.349898E-3
  fgMCL5S2S = 2.866516E-7

# set up the logger
var L = newConsoleLogger()
if not dirExists("logs"):
  createDir("logs")
var fL = newFileLogger("logs/mclimit.log", fmtStr = verboseFmtStr)
when isMainModule:
  addHandler(L)
  addHandler(fL)

proc clone*(h: Histogram): Histogram =
  ## performs a clone of a histogram
  result.ndim = h.ndim
  result.bins = h.bins.clone
  result.counts = h.counts.clone
  result.err = h.err.clone

proc clone*(ch: Channel): Channel =
  ## performs a clone of a channel by cloning all contained histograms
  result.systErr = ch.systErr # has value semantics
  result.sig = ch.sig.clone
  result.back = ch.back.clone
  result.cand = ch.cand.clone

proc clone*(data: DataSource): DataSource =
  ## clones a data source, by cloning all channels
  result = data.mapIt(it.clone)

proc getBins*(h: Histogram): int =
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
  const uni = uniform(0.0, 1.0)
  var
    a = 0.0
    b = 0.0
  while true:
    a = random(rnd, uni)
    b = (2.0 * random(rnd, uni) - 1.0) * K
    if  b * b <= -4.0 * a * a * ln(a): break
  result = mean + sigma * (b / a)

proc CLb*(cl: ConfidenceLevel, use_sMC: bool = false): float =
  ## Get the confidence limit for the background only
  if use_sMC:
    for idx in items(cl.iss):
      if cl.tss[idx] < cl.tsd:
        result += 1.0 / (cl.lrs[idx] * cl.nmc.float)
  else:
    var i = 0
    for idx in items(cl.isb):
      if cl.tsb[idx] < cl.tsd:
        # NOTE: so this is only the value with the highest tsb?
        result = (i + 1).float / cl.nmc.float
      inc i

proc CLsb*(cl: ConfidenceLevel, use_sMC: bool = false): float =
  ## Get the confidence limit for the signal plus background hypothesis
  if use_sMC:
    var i = 0
    for idx in items(cl.iss):
      if cl.tss[idx] <= cl.tsd:
        result = i.float / cl.nmc.float
      inc i
  else:
    for idx in items(cl.isb):
      if cl.tsb[idx] <= cl.tsd:
        result += cl.lrb[idx] / cl.nmc.float

proc CLs*(cl: ConfidenceLevel, use_sMC: bool = false): float =
  ## Get the confidence level defined by CLs = CLsb / CLb.
  ## This quantity is stable with respect to background fluctuations.
  let clb = cl.CLb(false) # NOTE: why `use_sMC` ignored here?
  let clsb = cl.CLsb(use_sMC)
  if clb == 0.0: warn "clb == 0!"
  else: result = clsb / clb

proc setTSB(cl: var ConfidenceLevel, tsb: Tensor[float]) =
  cl.tsb = tsb.clone
  cl.isb = tsb.argsort(SortOrder.Ascending)

proc setTSS(cl: var ConfidenceLevel, tss: Tensor[float]) =
  cl.tss = tss.clone
  cl.iss = tss.argsort(SortOrder.Ascending)

proc fluctuate(input: DataSource, output: var DataSource,
               rnd: var Random, stat: bool): bool =
  template statFluc(chIdx, field: untyped): untyped =
    var new = output[chIdx].field
    var old = input[chIdx].field
    if stat:
      # NOTE: I assume bin 0 is left out, because it's the underflow bin in ROOT
      for bin in 1 ..< new.getBins:
        let gaus = gaussian(0.0, old.err[bin])
        new.counts[bin] = old.counts[bin] + rnd.sample(gaus)
    else:
      new = old.clone
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
    let gausRnd = gaussian(0.0, 1.0)
    while reToss:
      var toss = zeros[float](input[0].systErr.len)
      toss.apply_inline(rnd.sample(gausRnd))
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

proc computeLimit*(data: DataSource, rnd: var Random,
                   stat: bool,
                   nmc: int = 1_000_000,
                   verbose = true
                  ): ConfidenceLevel =
  # determine the number of bins the channel with most bins has
  let nChannel = data.len
  let maxBins = max(data.mapIt(it.sig.getBins))
  template sumIt(fd: untyped): untyped = data.mapIt(it.fd.counts.sum).sum
  let nsig = sumIt(sig)
  let nbg = sumIt(back)
  let ncand = sumIt(cand)

  result = ConfidenceLevel(nmc: nmc, btot: nbg, stot: nsig, dtot: ncand)

  var fgTable = newTensor[float]([nChannel, maxbins])
  var buffer = 0.0
  for chIdx in 0 ..< nChannel:
    for bin in 0 ..< data[chIdx].sig.getBins:
      let s = data[chIdx].sig.counts[bin]
      let b = data[chIdx].back.counts[bin]
      let d = data[chIdx].cand.counts[bin]
      if almostEqual(b, 0.0) and s > 0.0:
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
    tmp1 = data.clone
    tmp2 = data.clone
    pois: Poisson
  for i in 0 ..< nmc:
    if i mod 5000 == 0 and verbose:
      echo "Iteration ", i
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
          elif s > 0.0 and b == 0.0:
            lrb[i] += 20.0 * rand - s
    lrs[i] = if lrs[i] < 710: exp(lrs[i]) else: exp(710.0)
    lrb[i] = if lrb[i] < 710: exp(lrb[i]) else: exp(710.0)

  result.setTSS tss
  result.setTSB tsb
  result.lrs = lrs
  result.lrb = lrb

proc getExpectedStatistic_b*(cl: ConfidenceLevel, sigma: range[-2 .. 2] = 0): float =
  template arg(cl, level): untyped =
    let inner = min(cl.nmc.float, max(1, (cl.nmc.float * level))).round.int
    let outer = cl.isb[inner]
    outer
  case sigma
  of -2:
    result = -2 * (cl.tsb[arg(cl, fgMCLP2S)] - cl.stot)
  of -1:
    result = -2 * (cl.tsb[arg(cl, fgMCLP1S)] - cl.stot)
  of 0:
    result = -2 * (cl.tsb[arg(cl, fgMCLMED)] - cl.stot)
  of 1:
    result = -2 * (cl.tsb[arg(cl, fgMCLM1S)] - cl.stot)
  of 2:
    result = -2 * (cl.tsb[arg(cl, fgMCLM2S)] - cl.stot)

proc getExpectedStatistic_sb*(cl: ConfidenceLevel, sigma: range[-2 .. 2] = 0): float =
  ## Get the expected statistic value in the signal plus background hypothesis
  template arg(cl, level): untyped =
    let inner = min(cl.nmc.float, max(1, (cl.nmc.float * level))).round.int
    let outer = cl.iss[inner]
    outer
  case sigma
  of -2:
    result = -2 * (cl.tss[arg(cl, fgMCLP2S)] - cl.stot)
  of -1:
    result = -2 * (cl.tss[arg(cl, fgMCLP1S)] - cl.stot)
  of 0:
    result = -2 * (cl.tss[arg(cl, fgMCLMED)] - cl.stot)
  of 1:
    result = -2 * (cl.tss[arg(cl, fgMCLM1S)] - cl.stot)
  of 2:
    result = -2 * (cl.tss[arg(cl, fgMCLM2S)] - cl.stot)

proc getExpectedCLsb_b*(cl: ConfidenceLevel, sigma: range[-2 .. 2] = 0): float =
  ## Get the expected Confidence Level for the signal plus background hypothesis
  ## if there is only background.
  template assignFor(level: untyped): untyped =
    for i in 0 ..< cl.nmc:
      let inner = min(cl.nmc.float, max(1, (cl.nmc.float * level))).round.int
      if (cl.tsb[cl.isb[i]] <= cl.tsb[cl.isb[inner]]):
        result += cl.lrb[cl.isb[i]] / cl.nmc.float
  case sigma
  of -2:
    assignFor(fgMCLP2S)
  of -1:
    assignFor(fgMCLP1S)
  of 0:
    assignFor(fgMCLMED)
  of 1:
    assignFor(fgMCLM1S)
  of 2:
    assignFor(fgMCLM2S)

proc getExpectedCLb_sb*(cl: ConfidenceLevel, sigma: range[-2 .. 2] = 0): float =
  ## Get the expected Confidence Level for the background only
  ## if there is signal and background.
  template assignFor(level: untyped): untyped =
    for i in 0 ..< cl.nmc:
      let inner = min(cl.nmc.float, max(1, (cl.nmc.float * level))).round.int
      if (cl.tss[cl.iss[i]] <= cl.tss[cl.iss[inner]]):
        result += cl.lrs[cl.iss[i]] / cl.nmc.float
  case sigma
  of -2:
    assignFor(fgMCLP2S)
  of -1:
    assignFor(fgMCLP1S)
  of 0:
    assignFor(fgMCLMED)
  of 1:
    assignFor(fgMCLM1S)
  of 2:
    assignFor(fgMCLM2S)

proc getExpectedCLb_b*(cl: ConfidenceLevel, sigma: range[-2 .. 2] = 0): float =
  ## Get the expected Confidence Level for the background only
  ## if there is only background.
  template assignFor(level: untyped): untyped =
    for i in 0 ..< cl.nmc:
      let inner = min(cl.nmc.float, max(1, (cl.nmc.float * level))).round.int
      if (cl.tsb[cl.isb[i]] <= cl.tsb[cl.isb[inner]]):
        result = (i + 1).float / cl.nmc.float
  case sigma
  of -2:
    assignFor(fgMCLP2S)
  of -1:
    assignFor(fgMCLP1S)
  of 0:
    assignFor(fgMCLMED)
  of 1:
    assignFor(fgMCLM1S)
  of 2:
    assignFor(fgMCLM2S)

proc getExpectedCLs_b*(cl: ConfidenceLevel, sigma: range[-2 .. 2] = 0): float =
  ## Get the expected CLs given the background
  result = cl.getExpectedCLsb_b(sigma) / cl.getExpectedCLb_b(sigma)
