import ../mclimit
import ggplotnim, seqmath, sequtils, arraymancer, tables

const nmc = 100_000
var rnd = wrap(initMersenneTwister(43))
let pois = poisson(5.0)
var
  sig: seq[float]
  back: seq[float]
  cand: seq[float]

proc poisson(k: int, λ: float): float =
  result = pow(λ, k.float) / fac(k).float * exp(-λ)
let x = arraymancer.arange(15)
let y = x.map_inline(seqmath.gauss(x.float, 7.0, 10.0))
for i in 0 ..< 10000:
  back.add rnd.sample(pois)
for i in 0 ..< 300:
  cand.add rnd.sample(pois)

var (backH, binsB) = histogram(back, range = (0.0, 15.0), bins = 15, density = true)
var (candH, binsC) = histogram(cand, range = (0.0, 15.0), bins = 15, density = true)
for i, x in binsC:
  candH[i] = candH[i]
  backH[i] = backH[i]

var df = seqsToDf({ "k" : x,
                    "poisson" : y,
                    "back" : backH,
                    "cand" : candH })
df.writeCsv("tpoisson.csv")

df = df.gather(["poisson", "back", "cand"], "Type", "y")
echo df
ggplot(df, aes(k, "y", color = "Type")) +
  geom_point() +
  ggsave("/tmp/poisson.pdf")

func toTensor[T](t: Tensor[T]): Tensor[T] = t
template toHisto(arg: typed): untyped =
  let counts = arg.toTensor.asType(float)
  # why cannot directly map_inline with sqrt :(
  let err = counts.toRawSeq.mapIt(it.sqrt).toTensor
  Histogram(ndim: 1,
            bins: backH.toTensor.asType(float),
            counts: counts,
            err: err)
let sigHist = toHisto(y)
let backHist = toHisto(backH)
let candHist = toHisto(candH)
let ch = Channel(sig: sigHist, back: backHist, cand: candHist,
                 systErr: { "Tel" : SystematicError(cand: 0.1, back: 0.1),
                            "Rad" : SystematicError(cand: 0.2, back: 0.25)}.toOrderedTable)
let limit = computeLimit(@[ch], rnd, stat = false, nmc = nmc)
echo limit.tsd
echo limit.tss.mean
echo "CLb: ", limit.CLb()
echo "CLsb: ", limit.CLsb()
echo "CLs: ", limit.CLs()
