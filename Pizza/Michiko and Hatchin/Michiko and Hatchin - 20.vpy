from vstools import core, vs, replace_ranges
core.max_cache_size = 32*1024
core.num_threads = 8


from awsmfunc import bbmod
from havsfunc import SMDegrain
from jvsfunc import retinex_edgemask
from lvsfunc import dpir
from rektlvls import rektlvls
from vodesfunc import Waifu2x_Doubler, NNEDI_Doubler
from vsdeband import F3kdb, adaptive_grain
from vskernels import Bilinear, BicubicDidee, Bicubic
from vsmasktools import MinMax
from vsscale import SSIM, descale_detail_mask
from vsutil import depth, get_y, iterate


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
opstart = 0 #first frame lights
opend = 1919

edstart = 30600
edend = 32519 #last frame all white



###SOURCE/FIX BORDERS
src = core.lsmas.LWLibavSource(r"B:\Michiko and Hatchin\MICHIKO_E_HATCHIN10\BDMV\STREAM\00000.m2ts")[32544:]

levels = rektlvls(depth(src, 16), colnum = [1,1918,1919], colval = [-4,-4,12], rownum = [0,1,1078,1079], rowval = [12,-4,-5,12])
levelsop = rektlvls(depth(src, 16), colnum = [1,1918,1919], colval = [-6,-4,12], rownum = [0,1,1078,1079], rowval = [13,-5,-5,12])
levelsed = rektlvls(depth(src, 16), colnum = [0,1,1918,1919], colval = [9,-11,-8,12], rownum = [0,1,1078,1079], rowval = [13,-9,-7,12])

levels = replace_ranges(levels, levelsop, [(opstart, opend)])
levels = replace_ranges(levels, levelsed, ([edstart, edend]))

fb = core.fb.FillBorders(levels, 1, 0, 0, 0, mode = "fillmargins")
bb = bbmod(fb, 0, 0, 2, 2, thresh = 4, blur = 20, u = True, v = True)
bb = bbmod(bb, 2, 2, 2, 2, thresh = 4, blur = 999, u = False, v = False)

bbop = bbmod(fb, 0, 0, 2, 2, thresh = 12, blur = 20, u = True, v = True)
bbop = bbmod(bbop, 2, 2, 2, 2, thresh = 3, blur = 999, u = False, v = False)

bbed = bbmod(fb, 2, 2, 2, 2, thresh = 6, blur = 80, u = False, v = False)
bb = replace_ranges(bb, bbop, [(opstart, opend)])


###RESCALE
src_y = get_y(depth(bb, 32))
descale = Bilinear().descale(src_y, 1280, 720)

w2x = Waifu2x_Doubler(tiles = 2, overlap = 8).double(descale)
sharprescale = SSIM(kernel=BicubicDidee).scale(w2x, 1920, 1080)

nnedi = NNEDI_Doubler(opencl = False).double(descale)
softrescale = SSIM(kernel = BicubicDidee).scale(nnedi, 1920, 1080)

mixed_upscale = core.akarin.Expr([nnedi, w2x], '0.2 x * 0.8 y * +')
mixed_down = Bicubic(b=0, c=0.5).scale(mixed_upscale, 1920, 1080)
blurry_down = Bicubic(b=0.25, c=0).scale(mixed_upscale, 1920, 1080)

#to explain what is going on here:
#there are text overlays in the show that are descalable, however the standard w2x rescale is so sharp that the downscale introduces a lot of ringing
#so we use a rescale that introduces it and one that doesnt to make a diffmask, and catch just those areas (works astoundingly well) and then apply
#a blurrier downscale to just those areas, and keep the sharper downscale that doesnt blur fine texture detail as much for the rest of the image

copemask = descale_detail_mask(sharprescale, softrescale, thr = 0.07, inflate = 1).std.Convolution([1]*25).std.Convolution([1]*25)
main_rescale = core.std.MaskedMerge(mixed_down, blurry_down, copemask)

rescale_spliced = replace_ranges(main_rescale, softrescale, [(opstart, opend)])
rescale_spliced = replace_ranges(rescale_spliced, src_y, [(edstart, edend)])
rescale_merged = core.std.ShufflePlanes([rescale_spliced, depth(bb, 32)], [0,1,2], vs.YUV)


###DEBLOCK + DENOISE
deblock = depth(dpir(rescale_merged, 7, cuda = 'trt', tiles = 2, fp16 = True, matrix = 1), 32)
preden = depth(SMDegrain(depth(deblock, 16), tr = 2, thSAD = 65, thSADC = 40, RefineMotion = True, prefilter = 2), 32)

bm3d = core.bm3dcuda_rtc.BM3D(clip = deblock, ref = preden, sigma = [0.60, 0.69, 0.69], radius = 3, block_step = 6, ps_range = 5).bm3d.VAggregate(radius = 3)
bm3d = core.bm3dcuda_rtc.BM3D(clip = deblock, ref = depth(bm3d, 32), sigma = [0.60, 0.69, 0.69], radius = 3, block_step = 5, ps_range = 6).bm3d.VAggregate(radius = 3)


###DEBAND
textures1 = texture_mask(get_y(bm3d), 1)
textures2 = texture_mask(get_y(bm3d), 5)

textures = core.std.Expr([textures1, textures2], 'x y max')
linemask = retinex_edgemask(bm3d, brz = 5700)

deband = F3kdb.deband(bm3d, 17, [32, 20, 20], 0)
debandmask = core.std.Expr([linemask, textures], 'x y max')
debandmasked = core.std.MaskedMerge(deband, bm3d, debandmask)


###GRAIN
grain1 = core.noise.Add(debandmasked, type = 3, var = 1.45, seed = 5, xsize = 2, ysize = 2, constant = True)
grain2 = core.noise.Add(grain1, type = 3, var = 2.2, seed = 5, xsize = 2, ysize = 2, constant = False)

grainmask = adaptive_grain(debandmasked, luma_scaling = 2, show_mask = True)
grain = core.std.MaskedMerge(grain1, grain2, grainmask)
ed_spliced = replace_ranges(grain, bbed, [(edstart, edend)])

###OUTPUT
out = depth(ed_spliced, 10)


out.set_output()
#src.set_output(2)
#kagura.set_output(3)