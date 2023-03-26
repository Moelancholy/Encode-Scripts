from vstools import  vs, core, depth, get_y

core.set_affinity(range(14, 24), 30000)

from havsfunc import SMDegrain
from vodesfunc import Waifu2x_Doubler, NNEDI_Doubler
from vsdehalo import fine_dehalo, dehalo_sigma
from vsdenoise import Prefilter
from vskernels import BicubicDidee
from vsscale import SSIM


###CUSTOM FUNCS
def eedi3s(clip, field=0, alpha=0.125, beta=0.25, gamma=50, nrad=2, mdis=20, vcheck=2, vthresh0=12, vthresh1=24, vthresh2=4, mclip=None): 
    out = clip.eedi3m.EEDI3(field, False, 0, alpha, beta, gamma, nrad, mdis, vcheck=vcheck, vthresh0=vthresh0, vthresh1=vthresh1, vthresh2=vthresh2, sclip=clip, mclip=mclip)
    if mclip is not None:
        return core.std.Expr([mclip, out, clip], 'x y z ?')
    return out


###GETTING SOURCES
src = depth(core.lsmas.LWLibavSource(r"A:\Do.It.Yourself.Vol2.2023.COMPLETE.BLURAY-KoushinRip\DO_IT_YOURSELF_2\BDMV\STREAM\00002.m2ts"), 16)


###AA
waifu = Waifu2x_Doubler(tiles=2).double(src)
nnedi = NNEDI_Doubler(False).double(src)
doubled = core.std.Expr([waifu, nnedi], "x 0.70 * y 0.30 * +")

ydown = SSIM(kernel=BicubicDidee).scale(doubled, 2880, 1620)
aa2 = eedi3s(ydown.std.Transpose(), gamma = 60)
aa2 = eedi3s(aa2.std.Transpose(), gamma = 60)
aa2 = SSIM(kernel=BicubicDidee).scale(aa2, 1920, 1080)

preden = SMDegrain(get_y(src), tr = 2, thSAD = 120, RefineMotion = True, prefilter = 2)
mask = preden.std.Prewitt().std.Binarize(65 << 8)
aamask = mask.std.Inflate()
aamasked = core.std.MaskedMerge(get_y(src), aa2, aamask)

grainmask1 = mask.std.Deflate().std.Deflate()
grainmask2 = core.std.Expr([aamask.std.Inflate(), grainmask1], "x y -").std.BoxBlur()

aagrain = core.grain.Add(aamasked, seed = 1, var = 0.70, uvar = 0.0, constant = True)
aagrained = core.std.MaskedMerge(aamasked, aagrain, grainmask2)


###DEHALO
doubledh = aagrained.znedi3.nnedi3(field=0, dh=True, nsize=4, nns=3).std.Transpose().znedi3.nnedi3(field=0, dh=True, nsize=4, nns=3).std.Transpose()
dehalomask = fine_dehalo(doubledh, show_mask=True).resize.Bicubic(1920, 1080)
dehalomask2 = dehalomask.std.BoxBlur()

dehalo = dehalo_sigma(doubledh, brightstr = 1, blur_func=Prefilter.GAUSSBLUR2(sigma = 1.65)).resize.Point(1920, 1080)
dehalo = core.std.MaskedMerge(aagrained, dehalo, dehalomask)

dehalograin = core.grain.Add(dehalo, seed = 1, var = 1.50, uvar = 0.0, constant = True)
dehalograined = core.std.MaskedMerge(dehalo, dehalograin, dehalomask2)


###CHROMA
merge = core.std.ShufflePlanes([dehalograined, src], [0,1,2], vs.YUV)
cden = core.knlm.KNLMeansCL(merge, d = 4, a = 4, s = 3, h = 0.25, channels = 'uv')


###OUTPUT
out = depth(cden, 10)
out.set_output()
#src.set_output(1)
#cden.set_output(2)
#grain.set_output(3)
#grainmask2.set_output(4)
#grainmask.set_output(5)