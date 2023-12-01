import vapoursynth as vs

from getfnative import descale_cropping_args
from jvsfunc import retinex_edgemask
from vodesfunc import Waifu2x_Doubler, NNEDI_Doubler
from vsaa import Eedi3SR
from vskernels import BicubicDidee
from vsmasktools import diff_rescale
from vsscale import SSIM
from vstools import get_y, depth, replace_ranges, get_w

core = vs.core



def eedi3s(clip, field=0, alpha=0.125, beta=0.25, gamma=50, nrad=2, mdis=20, vcheck=2, vthresh0=12, vthresh1=24, vthresh2=4, mclip=None): 
    out = clip.eedi3m.EEDI3(field, False, 0, alpha, beta, gamma, nrad, mdis, vcheck=vcheck, vthresh0=vthresh0, vthresh1=vthresh1, vthresh2=vthresh2, sclip=clip, mclip=mclip)
    if mclip is not None:
        return core.std.Expr([mclip, out, clip], 'x y z ?')
    return out


def basedAA(clip, height, gamma):
    double = NNEDI_Doubler().double(clip)
    down = SSIM(kernel=BicubicDidee).scale(double, width=get_w(height), height=height)
    eedi = eedi3s(down.std.Transpose(), gamma = gamma)
    eedi = eedi3s(eedi.std.Transpose(), gamma = gamma)
    out = SSIM(kernel=BicubicDidee).scale(double, width=1920, height=1080)
    return out




native_ranges = [
    (0, 25), (126, 138), (472, 1146)
]


strong_aa_ranges = [
    (94, 109), (126, 138), (400, 454),
    (1412, 1450), (1517, 1582), (1679, 1705),
    (1799, 1815), 
]

no_aa_ranges = [
    (2038, 2055)
]




###SOURCE
clip = core.lsmas.LWLibavSource(r"B:\[BDMV][アニメ] けいおん！\K_ON_BD_04\BDMV\STREAM\00003.m2ts")
clip_y = get_y(depth(clip, 32))


###RESCALE
d_args = descale_cropping_args(clip, src_height=714.75, base_height=720)
descale = core.descale.Debilinear(clip_y, **d_args)
upscale = Waifu2x_Doubler(cuda = True, tiles=2, overlap = 10).double(descale)

c_args = dict(src_width = d_args['src_width'] * 2, src_height = d_args['src_height'] * 2, src_left = d_args['src_left'] * 2, src_top = d_args['src_top'] * 2)
down = core.resize.Bicubic(upscale, 1920, 1080, **c_args)

linemask = retinex_edgemask(down)
rescale = core.std.MaskedMerge(clip_y, down, linemask)
rescale_fixed = depth(replace_ranges(rescale, clip_y, native_ranges), 16)


##AA
nnupscale = NNEDI_Doubler().double(rescale_fixed)
nndown1 = SSIM(kernel=BicubicDidee).scale(nnupscale, get_w(1620), 1620, sigmoid = True)
aamask = rescale_fixed.std.Prewitt().std.Binarize(65 << 8).std.Maximum().std.Convolution([1]*9)

aastrong = Eedi3SR(gamma = 100).aa(nndown1)
aastrong = SSIM(kernel=BicubicDidee).scale(aastrong, 1920, 1080, sigmoid = True)
aastrong = core.std.MaskedMerge(rescale_fixed, aastrong, aamask)

aaweak = eedi3s(nndown1.std.Transpose(), gamma=80)
aaweak = eedi3s(aaweak.std.Transpose(), gamma=80)
aaweak = SSIM(kernel=BicubicDidee).scale(aaweak, 1920, 1080, sigmoid = True)
aaweak = core.std.MaskedMerge(rescale_fixed, aaweak, aamask)

aa = replace_ranges(aaweak, aastrong, strong_aa_ranges)
aa = replace_ranges(aa, rescale_fixed, no_aa_ranges)






get_y(clip).set_output(1)
rescale_fixed.set_output(2)
aaweak.set_output(3)
aastrong.set_output(4)
#aa3.set_output(5)
#rescale.set_output(6)

#first screen and title screen dont descale
#wriggly stuff mask
#1911 dehlo, 2022


#edgeclean, dering(scenefilter?)
#denoise (test diff things chroma)
#deband weak, strong
#custom deband masks
#grain