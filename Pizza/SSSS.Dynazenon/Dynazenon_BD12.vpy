import vapoursynth as vs

import awsmfunc as awf
import debandshit as dbs
import kagefunc as kgf
import lvsfunc as lvf
import vardefunc as vdf
import vardefunc.mask as mask

from adptvgrnMod import adptvgrnMod
from functools import partial
from kagefunc import retinex_edgemask
from math import ceil
from havsfunc import SMDegrain
from muvsfunc import SSIM_downsample
from rekt import rektlvls
from thresholding import kimagure_thresholding
from vardefunc.scale import fsrcnnx_upscale, nnedi3cl_double, to_444
from vardefunc.sharp import z4usm
from vsutil import depth, fallback, get_y, get_depth, iterate, get_w

core = vs.core
core.max_cache_size = 42*1024
core.num_threads = 14


###CUSTOM FUNCS
def average_resize(clip, w=None, h=None):
    iw = clip.width
    ih = clip.height
    ow = fallback(w, iw)
    oh = fallback(h, ih)
    
    if (ow > iw and ow/iw != ow//iw) or (oh > ih and oh/ih != oh//ih):
        clip = clip.resize.Point(iw * ceil(ow / iw), ih * ceil(oh / ih))
    
    return clip.fmtc.resample(ow, oh, kernel='box', fulls=1, fulld=1)



def eedi3s(clip, field=0, alpha=0.125, beta=0.25, gamma=50, nrad=2, mdis=20, vcheck=2, vthresh0=12, vthresh1=24, vthresh2=4, mclip=None): 
    out = clip.eedi3m.EEDI3(field, False, 0, alpha, beta, gamma, nrad, mdis, vcheck=vcheck, vthresh0=vthresh0, vthresh1=vthresh1, vthresh2=vthresh2, sclip=clip, mclip=mclip)
    if mclip is not None:
        return core.std.Expr([mclip, out, clip], 'x y z ?')
    return out



def manual_masking(base: vs.VideoNode, mask_clip: vs.VideoNode, image_path: str, start, end, inflate = 3, debug = False):
    depth_base = get_depth(base)
    depth_clip = get_depth(mask_clip)
    if depth_base != mask_clip:
        if depth_base > depth_clip:
            depth_clip = depth_base
            mask_clip = depth(mask_clip, depth_clip)
        else:
            depth_base = depth_clip
            base = depth(base, depth_base)
    
    image = core.imwri.Read(image_path)
    image = image * base.num_frames
    image = get_y(core.resize.Bicubic(image, format = vs.YUV420P16, matrix = 1))
    image = depth(image, depth_base)
    image = iterate(image, core.std.Inflate, inflate)

    if debug == False:
        masked = core.std.MaskedMerge(base, mask_clip, image)
        splice = base[:start] + masked[start:end+1] + base[end+1:]
    
        return splice
    else:
        return image



def merged_doubler(clip: vs.VideoNode, shader: str, b = 1, c = 0, fsrcnnx_str = 0.7, nnedi_str = 0.3, bicubic_str = 0.0, radius = 2, sharpen = 65):
    total = float(fsrcnnx_str + nnedi_str + bicubic_str)
    fstr = float(fsrcnnx_str/total)
    nstr = float(nnedi_str/total)
    bstr = float(bicubic_str/total)

    fsrcnnx = fsrcnnx_upscale(clip, clip.width *2, clip.height *2, shader_file = shader, downscaler = None, profile = 'zastin', sharpener = partial(z4usm, radius = radius, strength = sharpen))
    nnedi = nnedi3cl_double(clip, pscrn = 1)
    bicubic = core.resize.Bicubic(clip, clip.width * 2, clip.height * 2, filter_param_a = b, filter_param_b = c)

    merge = core.std.Expr([fsrcnnx, nnedi, bicubic], f"{fstr} x * {nstr} y * + {bstr} z * +")
    return merge



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
    rmask = mask.MinMax(range).get_mask(clip, lthr=0, multi=1.00)
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



def CCD_denoise(clip: vs.VideoNode, strength):
    depth_clip = get_depth(clip)
    clip1 = to_444(clip, width = clip.width, height = clip.height, join_planes = True)
    clip1 = core.resize.Bicubic(clip1, matrix_in = 1, primaries_in = 1, transfer_in = 1, matrix = 1, primaries = 1, transfer = 1)
    clip1 = core.resize.Bicubic(clip1, format = vs.RGBS)
    clip1 = core.ccd.CCD(clip1, strength)
    clip1 = core.resize.Bicubic(clip1, format = vs.YUV420P16, matrix = 1)
    clip1 = depth(clip1, depth_clip)
    clip = core.std.ShufflePlanes([clip, clip1], [0, 1, 2], vs.YUV)
    return clip


def manual_masking(base: vs.VideoNode, mask_clip: vs.VideoNode, image_path: str, start, end, inflate = 3, debug = False):
    depth_base = get_depth(base)
    depth_clip = get_depth(mask_clip)
    if depth_base != mask_clip:
        if depth_base > depth_clip:
            depth_clip = depth_base
            mask_clip = depth(mask_clip, depth_clip)
        else:
            depth_base = depth_clip
            base = depth(base, depth_base)
    
    image = core.imwri.Read(image_path)
    image = image * base.num_frames
    image = get_y(core.resize.Bicubic(image, format = vs.YUV420P16, matrix = 1))
    image = depth(image, depth_base)
    image = iterate(image, core.std.Inflate, inflate)

    if debug == False:
        masked = core.std.MaskedMerge(base, mask_clip, image)
        splice = base[:start] + masked[start:end+1] + base[end+1:]
    
        return splice
    else:
        return image




###SCENEFILTERING
edstart = 31887 #first frame washing machine
edend = 34044 #last frame trigger
nced_length = edend - edstart + 1

opstart = 1 #first frame emblem
opend = 2157 #last frame dragon
ncop_length = opend - opstart +2

native_elements = [
    (20907, 22413), (22606, 24415), (33960, 34045)
    ]

MOVES_LIKE_JAGGED_RANGES = [
    (2775, 2849), (6043, 6131), (20444, 20509),
    (27874, 27960)
]

deband_ranges = [
    (3387, 3662), (3777, 3890), (4200, 4397),
    (4800, 4886), (8040, 8135), (13951, 13989),
    (14982, 15024), (15886, 16129), (20210, 20371),
    (28273, 28434)
]


nuke_ranges = [
    
]


###SOURCES
core = vs.core

src = core.lsmas.LWLibavSource(r"E:\Anime\[BDMV][SSSS.DYNAZENON][Vol.1-4]\[PCXP-50864][SSSS_DYNAZENON_4]\BDMV\STREAM\00002.m2ts")
ncop = core.lsmas.LWLibavSource(r"E:\Anime\[BDMV][SSSS.DYNAZENON][Vol.1-4]\[PCXP-50864][SSSS_DYNAZENON_4]\BDMV\STREAM\00011.m2ts")[:ncop_length]
nced = core.lsmas.LWLibavSource(r"E:\Anime\[BDMV][SSSS.DYNAZENON][Vol.1-4]\[PCXP-50862][SSSS_DYNAZENON_2]\BDMV\STREAM\00010.m2ts")[:nced_length]
src16 = depth(src, 16)
shader = r"C:\Users\twint\AppData\Roaming\mpv\shaders\FSRCNNX_x2_56-16-4-1.glsl"


###FIXBORDERS
fixb = rektlvls(src16, colnum = [0, 1919], colval = [16, 16], rownum = [0, 1079], rowval = [17, 17])
bb = awf.bbmod(fixb, 1, 1, 1, 1, thresh = 5, blur = 999, u = False, v = False)
bb8 = depth(bb, 8)

finalbb = bb
bb32 = depth(finalbb, 32)


###RESCALE + AA
descale = core.descale.Debicubic(get_y(bb32), width = 1280, height = 720, b = 0, c = 1/2)
rescale = merged_doubler(descale, shader = shader, fsrcnnx_str = 1, nnedi_str = 0.45, sharpen = 35)

mediumaa = eedi3s(rescale.std.Transpose(), gamma = 70)
mediumaa = eedi3s(mediumaa.std.Transpose(), gamma = 70)
mediumaa = core.resize.Bicubic(mediumaa, format = vs.RGBS)
mediumaa = SSIM_downsample(mediumaa, w = 1920, h = 1080, sigmoid = True, filter_param_a = -1/2, filter_param_b = 1/4)
mediumaa = get_y(core.resize.Bicubic(mediumaa, format = vs.YUV420P16, matrix = 1))

strongaa = lvf.sraa(rescale, 1, rescale.width, rescale.height, downscaler = None)
strongaa = core.resize.Bicubic(strongaa, format = vs.RGBS)
strongaa = SSIM_downsample(strongaa, w = 1920, h = 1080, sigmoid = True, filter_param_a = -1/2, filter_param_b = 1/4)
strongaa = get_y(core.resize.Bicubic(strongaa, format = vs.YUV420P16, matrix = 1))

down = core.resize.Bicubic(rescale, format = vs.RGBS)
down = SSIM_downsample(down, w = 1920, h = 1080, sigmoid = True, filter_param_a = -1/2, filter_param_b = 1/4)
down = get_y(core.resize.Bicubic(down, format = vs.YUV420P16, matrix = 1))
rescalemask = retinex_edgemask(down)
down = core.std.MaskedMerge(get_y(bb), down, rescalemask)

aamask = down.std.Prewitt().std.Binarize(60 << 8).std.Maximum().rgvs.RemoveGrain(20)
aa = lvf.rfs(mediumaa, strongaa, MOVES_LIKE_JAGGED_RANGES)
aamasked = core.std.MaskedMerge(down, aa, aamask)


###ERROR MASK
upscale = core.resize.Bicubic(descale, 1920, 1080)
error_mask = lvf.scale.descale_detail_mask(clip = get_y(bb32), rescaled_clip = upscale)
error_mask = error_mask.std.Maximum().std.Inflate().std.Maximum().std.Inflate().std.Maximum().std.Inflate().std.Maximum().std.Inflate().std.Convolution([1]*25)

error_masked = core.std.MaskedMerge(aamasked, get_y(bb), depth(error_mask, 16))
error_splice = lvf.rfs(aamasked, error_masked, native_elements)


###DENOISE
preden = SMDegrain(depth(error_splice, 16), tr = 2, thSAD = 150, RefineMotion = True, prefilter = 2)
tmask = texture_mask(preden, range = 1)

denoise_dark = core.bm3dcpu.BM3D(clip = depth(error_splice, 32), ref = depth(preden, 32), sigma = 0.60, radius = 2).bm3d.VAggregate(radius = 2)
denoise_dark = core.bm3dcpu.BM3D(clip = depth(error_splice, 32), ref = depth(denoise_dark, 32), sigma = 0.60, radius = 2).bm3d.VAggregate(radius = 2)

denoise_bright = core.bm3dcpu.BM3D(clip = depth(error_splice, 32), ref = depth(preden, 32), sigma = 0.88, radius = 2).bm3d.VAggregate(radius = 2)
denoise_bright = core.bm3dcpu.BM3D(clip = depth(error_splice, 32), ref = depth(denoise_bright, 32), sigma = 0.88, radius = 2).bm3d.VAggregate(radius = 2)

luma_mask = core.adg.Mask(bb.std.PlaneStats(), luma_scaling=8)
denoise_luma = core.std.MaskedMerge(denoise_bright, denoise_dark, luma_mask)

cden1 = CCD_denoise(bb, 3)
cden2 = CCD_denoise(bb, 5)

cden = kimagure_thresholding(bb, cden1, cden2)
cden = core.std.MaskedMerge(cden, bb, tmask)
denoise = core.std.ShufflePlanes([denoise_luma, cden], [0, 1, 2], vs.YUV)


###DEBAND
weakdeband = dbs.dumb3kdb(denoise, radius = 17, threshold = [32, 28, 28], grain = 0)
weakdeband = adptvgrnMod(weakdeband, strength = 0.06, static = True, luma_scaling = 7, grain_chroma = False, size = 1.2, sharp = 75)
weakdeband = adptvgrnMod(weakdeband, strength = 0.03, static = False, luma_scaling = 7, grain_chroma = False, size = 1.2, sharp = 75)

strongdeband = dbs.dumb3kdb(denoise, radius = 19, threshold = [48, 40, 40], grain = 0)
strongdeband = adptvgrnMod(strongdeband, strength = 0.08, static = True, luma_scaling = 7, grain_chroma = False, size = 1.2, sharp = 75)
strongdeband = adptvgrnMod(strongdeband, strength = 0.05, static = False, luma_scaling = 7, grain_chroma = False, size = 1.2, sharp = 75)

nuke = dbs.placebo_deband(denoise, radius = 30, threshold=[10, 10, 10], iterations = 2, grain = 0)
nuke = adptvgrnMod(nuke, strength = 0.08, static = True, luma_scaling = 7, grain_chroma = False, size = 1.2, sharp = 75)
nuke = adptvgrnMod(nuke, strength = 0.05, static = False, luma_scaling = 7, grain_chroma = False, size = 1.2, sharp = 75)

deband = lvf.rfs(weakdeband, strongdeband, deband_ranges)
deband = lvf.rfs(deband, nuke, nuke_ranges)

debandmask = core.std.Expr([rescalemask, tmask], 'x y +')
debandmasked = core.std.MaskedMerge(deband, denoise, debandmask)


###OUTPUT
grain = adptvgrnMod(debandmasked, strength = 0.06, static = True, luma_scaling = 2, grain_chroma = False, size = 1.2, sharp = 75)
grain = adptvgrnMod(grain, strength = 0.11, static = False, luma_scaling = 10, grain_chroma = False, size = 1.2, sharp = 75)
out = depth(grain, 10)

out.set_output()
src.set_output(1)
#mediumaa.set_output(2)
#strongaa.set_output(3)
#weakdeband.set_output(4)
#strongdeband.set_output(5)
#error_mask.set_output(6)
#nuke.set_output(7)

#0,132,b=1.2/187,222,b=1.3/661,960,b=1.2/961,1065,b=1.3/1489,1578,b=1.1/1663,1710,b=1.1/1738,1827,b=1.3/1975,2052,b=1.2/2385,2687,b=1.1/2688,2774,b=1.3/2775,2849,b=1.1/3891,4199,b=1.1/4458,4799,b=1.1/4887,4906,b=1.2/4976,5270,b=1.2/5271,5403,b=1.3/5569,5775,b=1.1/5848,5935,b=1.3/6043,6131,b=1.4/6486,7618,b=1.1/7679,7759,b=1.3/7848,8544,b=1.1/8687,8803,b=1.1/8804,8959,b=1.3/9253,9412,b=1.3/9556,9633,b=1.1/9634,9681,b=1.3/9838,9873,b=1.1/10840,12048,b=1.2/12049,13971,b=1.3/14182,14532,b=1.2/15147,15623,b=1.3/15696,17423,b=1.3/17485,17605,b=1.3/17864,18525,b=1.3/20835,25117,b=1.1/29931,29978,b=1.3/