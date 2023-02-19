from vstools import replace_ranges, get_w, depth, get_y, iterate, vs, core

core.set_affinity(range(12, 24), 40000)

from havsfunc import SMDegrain
from jvsfunc import retinex_edgemask
from lvsfunc import dpir
from vodesfunc import NNEDI_Doubler
from vsdeband import F3kdb, adaptive_grain
from vsdehalo import fine_dehalo
from vskernels import BicubicDidee
from vsmasktools import MinMax
from vsrgtools import lehmer_diff_merge
from vsscale import SSIM




def eedi3s(clip, field=0, alpha=0.125, beta=0.25, gamma=50, nrad=2, mdis=20, vcheck=2, vthresh0=12, vthresh1=24, vthresh2=4, mclip=None): 
    out = clip.eedi3m.EEDI3(field, False, 0, alpha, beta, gamma, nrad, mdis, vcheck=vcheck, vthresh0=vthresh0, vthresh1=vthresh1, vthresh2=vthresh2, sclip=clip, mclip=mclip)
    if mclip is not None:
        return core.std.Expr([mclip, out, clip], 'x y z ?')
    return out


def LevelsM(clip, points, levels, xpass=[0, 'peak'], return_expr=False):
    qm = len(points)
    peak = [(1 << clip.format.bits_per_sample) - 1, 1][clip.format.sample_type]
   
    if len(set(xpass)) == 1:
        expr = f'x {points[0]} < x {points[-1]} > or {xpass[0]} '
        qm -= 1
    else:
        expr = f'x {points[0]} < {xpass[0]} x {points[-1]} > {xpass[-1]} '
   
    for x in range(len(points) - 1):
        if points[x+1] < points[-1]:
            expr += f' x {points[x+1]} <= '
        if levels[x] == levels[x+1]:
            expr += f' {peak * levels[x]} '
        else:
            expr += f' x {points[x]} - {peak * (levels[x+1] - levels[x])/(points[x+1] - points[x])} * {peak * levels[x]} + '
   
    for _ in range(qm):
        expr += ' ? '
   
    expr = expr.replace('  ', ' ').replace('peak', f'{peak}')
   
    if return_expr:
        return expr
   
    return clip.std.Expr(expr)


def texture_mask(clip, range):
    ed_gray = core.std.ShufflePlanes(clip, 0, vs.GRAY)
    rmask = MinMax(range).edgemask(clip, lthr=0, multi=1.00)
    emask = ed_gray.std.Prewitt()
    em_hi = emask.std.Binarize(60 * 257, v0=65535, v1=0)
    em_hi = iterate(em_hi, core.std.Minimum, 5)
    em_me = emask.std.Binarize(40 * 257, v0=65535, v1=0)
    em_me = iterate(em_me, core.std.Minimum, 4)
    em_lo = emask.std.Binarize(20 * 257, v0=65535, v1=0)
    em_lo = iterate(em_lo, core.std.Minimum, 2)
    rm_txt = core.std.Expr([rmask, em_hi, em_me, em_lo], 'x y z a min min min')
    weighted = LevelsM(rm_txt, points=[x *256 for x in (1.75, 2.5, 5, 10)], levels=[0, 1, 1, 0], xpass=[0,0], return_expr=0)

    masked = weighted.std.BoxBlur(hradius=8,vradius=8).std.Expr(f'x {65535 * 0.2} - {1 / (1 - 0.2)} *')
    return masked


###SCENEFILTER
ed_ranges = [
    (31886, 34044)
]
#end = last frame before completely faded


hidive_frames = [
    6807,12931,13207,
]


###SOURCE
timescbr = core.lsmas.LWLibavSource(r"C:\Users\twint\Downloads\Kubo-san wa Mob wo Yurusanai - 06 (Amazon AnimeTimes CBR 1080p).mkv")
timescbr = core.std.FreezeFrames(timescbr, first = [6095,6927,8187,10679,11599,11695,12367,12655,12823,13491,16671,20363,23447,23471,24127,24987,26495,28503,30815,31187], last= [6095,6927,8187,10679,11599,11695,12367,12655,12823,13491,16671,20363,23447,23471,24127,24987,26495,28503,30815,31187], replacement=[6096,6928,8188,10680,11600,11696,12368,12656,12824,13492,16672,20364,23448,23472,14128,24988,26496,26504,30816,31188])

timesvbr = core.lsmas.LWLibavSource(r"C:\Users\twint\Downloads\Kubo-san wa Mob wo Yurusanai - 06 (Amazon AnimeTimes VBR 1080p).mkv")
timesvbr = core.std.FreezeFrames(timesvbr, first = [6095,6927,8187,10679,11599,11695,12367,12655,12823,13491,16671,20363,23447,23471,24127,24987,26495,28503,30815,31187], last= [6095,6927,8187,10679,11599,11695,12367,12655,12823,13491,16671,20363,23447,23471,24127,24987,26495,28503,30815,31187], replacement=[6096,6928,8188,10680,11600,11696,12368,12656,12824,13492,16672,20364,23448,23472,14128,24988,26496,26504,30816,31188])

hidive = core.lsmas.LWLibavSource(r"B:\[SubsPlease] Kubo-san wa Mob wo Yurusanai - 06 (1080p) [1D9D1A4B].mkv")[24:34069]
lehmer = depth(lehmer_diff_merge([timesvbr, timescbr, hidive]), 16)
lehmer = replace_ranges(lehmer, depth(hidive, 16), hidive_frames)



#6095,6927,8187,10679,11599,11695,12367,12655,12823,13491,16671,20363,23447,23471,24127,24987,26495,28503,30815,31187
#6096,6928,8188,10680,11600,11696,12368,12656,12824,13492,16672,20364,23448,23472,14128,24988,26496,26504,30816,31188

#31187


###AA
upscale = NNEDI_Doubler().double(lehmer)
midscale = SSIM(kernel=BicubicDidee).scale(upscale, get_w(1620), 1620)

aa2 = eedi3s(midscale.std.Transpose(), gamma = 80)
aa2 = eedi3s(aa2.std.Transpose(), gamma = 80)
aa2 = SSIM(kernel=BicubicDidee).scale(aa2, 1920, 1080)


###DERING
dehalo = fine_dehalo(aa2, rx = 2.3, brightstr=1.0, pre_ss = 2)
dehalo_merged = core.std.ShufflePlanes([dehalo, depth(lehmer, 16)], [0,1,2], vs.YUV)
deblock = depth(dpir(dehalo_merged, 50, cuda = 'trt', tiles = 2, fp16 = True), 32)


###DENOISE
preden = depth(SMDegrain(depth(deblock, 16), tr = 2, thSAD = 150, RefineMotion = True, prefilter = 2), 32)
denoise = core.bm3dcuda_rtc.BM3D(clip = deblock, ref = preden, sigma = 0.75, radius = 2).bm3d.VAggregate(radius = 2)
denoise = core.bm3dcuda_rtc.BM3D(clip = deblock, ref = depth(denoise, 32), sigma = 0.75, radius = 2).bm3d.VAggregate(radius = 2)


###DEBAND
linemask = retinex_edgemask(denoise)
textures = texture_mask(get_y(denoise), 1)
debandmask = core.akarin.Expr([linemask, textures], 'x 1.5 y * +')

deband = F3kdb.deband(denoise, 17, [25, 30, 30], 0)
deband = core.std.MaskedMerge(deband, denoise, debandmask)

grainmask = adaptive_grain(deband, luma_scaling = 4, show_mask = True)

grain1 = core.noise.Add(deband, type = 3, var = 2, seed = 5)
grain1 = core.std.MaskedMerge(deband, grain1, grainmask)
grain2 = core.noise.Add(depth(hidive, 16), type = 2, var = 3, seed = 5)
grain = replace_ranges(grain1, grain2, ed_ranges)


###OUTPUT
out = depth(grain, 10)
out.set_output()


#hidive.set_output(1)
#timescbr.set_output(2)
#lehmer.set_output(3)
#grain3.set_output(4)
#grain1.set_output(5)
#grain2.set_output(6)
#grain3.set_output(7)