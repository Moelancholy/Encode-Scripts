
import vapoursynth as vs

import debandshit as dbs
import lvsfunc as lvf
import vardefunc.mask as mask

from adptvgrnMod import adptvgrnMod, sizedgrn
from functools import partial
from kagefunc import retinex_edgemask
from lvsfunc.scale import descale_detail_mask
from havsfunc import SMDegrain
from math import ceil
from muvsfunc import SSIM_downsample
from thresholding import thresholding, Thresholds
from vardefunc.scale import fsrcnnx_upscale, nnedi3cl_double, to_444
from vardefunc.sharp import z4usm
from vsutil import get_depth, get_w, get_y, depth, iterate, fallback

core = vs.core



###CUSTOM FUNCS
def average_resize(clip, w=None, h=None):
    iw = clip.width
    ih = clip.height
    ow = fallback(w, iw)
    oh = fallback(h, ih)
    
    if (ow > iw and ow/iw != ow//iw) or (oh > ih and oh/ih != oh//ih):
        clip = clip.resize.Point(iw * ceil(ow / iw), ih * ceil(oh / ih))
    
    return clip.fmtc.resample(ow, oh, kernel='box', fulls=1, fulld=1)



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






###SCENEFILTER
opstart = 864 
edstart = 30882 
guest_drawing = 33927
eyecatch = 14650

strong_deband_ranges = [
(27379, 27462), (28555, 28571)
]

MOVES_LIKE_JAGGED_RANGES = [
(0, 52), (210, 256), (4138, 4323),
(8344, 8750), (9833, 9945), (12466, 12576),
(12955, 13041), (19041, 19316), (24004, 24027),
(27258, 27330), (27379, 27462), (27511, 27558),
(28077, 28292), (28406, 28571), (28925, 29185),
(30337, 30477)
]



###IMPORT SOURCES
src = depth(core.lsmas.LWLibavSource(r"E:\Anime\[SubsPlease] Honzuki no Gekokujou - 28 (1080p) [B6EACF9D].mkv").std.AssumeFPS(fpsnum = 24000, fpsden = 1001), 32)
src_y = get_y(src)
src16 = depth(src_y, 16)
shader = r"C:\Users\twint\AppData\Roaming\mpv\shaders\FSRCNNX_x2_56-16-4-1.glsl"

openingbook = r"C:\Users\twint\Documents\GitHub\Encode-Scripts\GJM\Ascendance of a Bookworm S3 (WEB)\masks\opening_book1.png"
titlebook = r"C:\Users\twint\Documents\GitHub\Encode-Scripts\GJM\Ascendance of a Bookworm S3 (WEB)\masks\titlecard_book.png"
titletext1 = r"C:\Users\twint\Documents\GitHub\Encode-Scripts\GJM\Ascendance of a Bookworm S3 (WEB)\masks\titlecard_title.png"
titletext2 = r"C:\Users\twint\Documents\GitHub\Encode-Scripts\GJM\Ascendance of a Bookworm S3 (WEB)\masks\titlecard_nextep.png"
eyecatch1 = r"C:\Users\twint\Documents\GitHub\Encode-Scripts\GJM\Ascendance of a Bookworm S3 (WEB)\masks\eyecatch1.png"
eyecatch2 = r"C:\Users\twint\Documents\GitHub\Encode-Scripts\GJM\Ascendance of a Bookworm S3 (WEB)\masks\eyecatch2.png"
darkhouse = r"C:\Users\twint\Documents\GitHub\Encode-Scripts\GJM\Ascendance of a Bookworm S3 (WEB)\masks\dark house.png"



###DESCALE
descale = core.descale.Debicubic(src_y, get_w(812), 812, b = 0, c = 0.5)
errorscale = core.resize.Bicubic(descale, 1920, 1080)
errormask = descale_detail_mask(src_y, errorscale)
errormask = errormask.resize.Bicubic(640, 360)

errormask = iterate(errormask, core.std.Maximum, 8)
errormask = iterate(errormask, core.std.Inflate, 5).resize.Bicubic(1920, 1080)

###DENOISE LUMA
preden2 = SMDegrain(depth(descale, 16), tr = 2, thSAD = 150, RefineMotion = True, prefilter = 2)
denoise3 = core.bm3dcpu.BM3D(clip = descale, ref = depth(preden2, 32), sigma = 0.69, radius = 2).bm3d.VAggregate(radius = 2)
denoise3 = core.bm3dcpu.BM3D(clip = descale, ref = depth(denoise3, 32), sigma = 0.69, radius = 2).bm3d.VAggregate(radius = 2)

sloc = [0, 0.5, 0.3, 3, 0.7, 6, 1, 8]
preden4 = core.dfttest.DFTTest(depth(descale, 16), sigma=1, ssx=sloc, ssy=sloc, tbsize=3, opt=0)


###RESCALE + AA
preden1 = preden2.bilateral.Bilateral(sigmaS = 2, sigmaR = 4/255)
aamask1 = mask.FreyChen().get_mask(preden1)
aamask2 = mask.FreyChen().get_mask(preden1).std.Binarize(40<<7)
aamask3 = mask.FreyChen().get_mask(preden1).std.Binarize(80<<7)
aamaskweak = core.std.Expr([aamask2, aamask3, aamask1], f"x 0.8 * y + 1.8 / 0.3 z * +").std.Maximum().std.Inflate().std.Inflate()
aamaskweak = average_resize(aamaskweak, src.width, src.height)

rescale = merged_doubler(denoise3, shader, fsrcnnx_str = 0.62, nnedi_str = 0.38, radius = 2, sharpen = 40)
strongaa = lvf.sraa(rescale, 1, rescale.width, rescale.height, downscaler = None)
strongaa = core.resize.Bicubic(strongaa, format = vs.RGBS)
strongaa = SSIM_downsample(strongaa, w = 1920, h = 1080, sigmoid = True)
strongaa = get_y(core.resize.Bicubic(strongaa, format = vs.YUV420P16, matrix = 1))

down = core.resize.Bicubic(rescale, format = vs.RGBS)
down = SSIM_downsample(down, w = 1920, h = 1080, sigmoid = True, filter_param_a = -1/2, filter_param_b = 1/4)
down = get_y(core.resize.Bicubic(down, format = vs.YUV420P16, matrix = 1))

nnedi = lvf.aa.taa(down, lvf.aa.nnedi3(opencl=True))
aa1 = lvf.aa.clamp_aa(down, nnedi, strongaa, 0.7)
aa1 = core.std.MaskedMerge(down, aa1, aamaskweak)

aa = lvf.rfs(aa1, strongaa, [opstart+119, opstart+195])
aa = manual_masking(aa, strongaa, start = opstart+195, end = opstart+261, image_path = openingbook)
aa = manual_masking(aa, strongaa, start = opstart+2158, end = opstart+2277, image_path = titlebook)
aa = manual_masking(aa, src16, start = opstart+2158, end = opstart+2277, image_path = titletext1)
aa = manual_masking(aa, src16, start = edstart+2157, end = edstart+2229, image_path = titletext2)
aa = lvf.rfs(aa, strongaa, MOVES_LIKE_JAGGED_RANGES)


###MASK CREDITS ETC
rescale_masked1 = manual_masking(aa, src16, start = eyecatch, end = eyecatch+94, inflate = 3, debug = False, image_path = eyecatch1)
rescale_masked1 = manual_masking(rescale_masked1, src16, start = eyecatch+145, end = eyecatch+239, image_path = eyecatch2)
rescale_masked2 = core.std.MaskedMerge(rescale_masked1, depth(src_y, 16), depth(errormask, 16))
rescale_masked = rescale_masked1[:opstart] + rescale_masked2[opstart:opstart+2158] + rescale_masked1[opstart+2158:edstart] + src16[edstart:edstart+2157] + rescale_masked1[edstart+2157:guest_drawing] + src16[guest_drawing:]


###MASKS AND DENOISE CHROMA
preden3 = core.resize.Bicubic(preden4, 1920, 1080)
linemask1 = retinex_edgemask(preden3, 1)
linemask2 = lvf.mask.detail_mask(preden3, brz_a=0.02, brz_b=0.015)
linemask3 = texture_mask(preden3, 2)
linemask4 = texture_mask(preden3, 3)

chromadn = depth(CCD_denoise(src, 4), 16)
chroma_mask = core.std.Expr([linemask1, linemask3 , linemask4], '0.7 x * 0.7 x * + 0.7 z * +')
denoise_masked_chroma = core.std.MaskedMerge(chromadn, depth(src, 16), chroma_mask)
denoise_masked = core.std.ShufflePlanes([rescale_masked, denoise_masked_chroma], [0, 1, 2], vs.YUV)


###DEBAND
debandmask = core.std.Expr([linemask1, linemask2, linemask4], 'x 0.4 y * + 0.6 z * +')
deband = dbs.dumb3kdb(denoise_masked, radius = 17, threshold = [30, 25, 25], grain = 10)
deband_masked = core.std.MaskedMerge(deband, denoise_masked, debandmask)

strong_deband = dbs.dumb3kdb(denoise_masked, radius = 20, threshold = [80, 70, 70], grain = 15)
strong_deband = adptvgrnMod(strong_deband, strength = 0.30, size = 1.2, sharp = 60, luma_scaling = 14, static = True)
strong_deband = adptvgrnMod(strong_deband, strength = 0.30, size = 1.2, sharp = 60, luma_scaling = 6, static = False)

medium_deband = dbs.dumb3kdb(denoise_masked, radius = 17, threshold = [48, 36, 36], grain = 10)
medium_deband = adptvgrnMod(medium_deband, strength = 0.15, size = 1.2, sharp = 60, luma_scaling = 14, static = True)
medium_deband = adptvgrnMod(medium_deband, strength = 0.15, size = 1.2, sharp = 60, luma_scaling = 6, static = False)
strong_deband_masked = core.std.MaskedMerge(medium_deband, denoise_masked, debandmask)

#deband_masked = manual_masking(deband_masked, strong_deband, start = 16422, end = 16481, image_path = darkhouse)
#deband_masked = manual_masking(deband_masked, strong_deband, start = 28974, end = 29021, image_path = darkhouse)
debandmasked = lvf.rfs(deband_masked, strong_deband_masked, strong_deband_ranges)



###GRAIN
dynamicgrain = adptvgrnMod(deband_masked, strength = 0.30, size = 1.2, sharp = 60, static = False, luma_scaling = 6, grain_chroma = False)
staticgrain = sizedgrn(deband_masked, strength = 0.06, size = 1.2, sharp = 60, static = True, grain_chroma = False)

thrs_a = tuple([0.35, 0.60, 1, 1])
thrs_b = tuple([0.00, 0.00, 0.35, 0.60])

mergegrain = thresholding(
        Thresholds(get_y(staticgrain), *thrs_a, None, None),
        Thresholds(get_y(dynamicgrain), *thrs_b, None, None),
        base = get_y(deband_masked)
)


###OUTPUT
grain = core.std.ShufflePlanes([mergegrain, deband_masked], [0, 1, 2], vs.YUV)
out = depth(grain, 10)


#src.set_output(1)
#deband_masked.set_output(2)
out.set_output()
#aa.set_output(2)
#strongaa.set_output(3)
#deband_masked.set_output(4)
#strong_deband_masked.set_output(5)
