import vapoursynth as vs
core = vs.core

core.max_cache_size = 48*1024
core.num_threads = 12

import debandshit as dbs
import lvsfunc as lvf
import vardefunc as vdf
import vardefunc.mask as mask

from adptvgrnMod import sizedgrn
from kagefunc import retinex_edgemask
from havsfunc import SMDegrain, FineDehalo, EdgeCleaner
from muvsfunc import SSIM_downsample
from vardefunc.scale import nnedi3cl_double
from vsutil import depth, get_y, get_depth, iterate, get_w, plane, split, join


###CUSTOM FUNCS
def eedi3s(clip, field=0, alpha=0.125, beta=0.25, gamma=50, nrad=2, mdis=20, vcheck=2, vthresh0=12, vthresh1=24, vthresh2=4, mclip=None): 
    out = clip.eedi3m.EEDI3(field, False, 0, alpha, beta, gamma, nrad, mdis, vcheck=vcheck, vthresh0=vthresh0, vthresh1=vthresh1, vthresh2=vthresh2, sclip=clip, mclip=mclip)
    if mclip is not None:
        return core.std.Expr([mclip, out, clip], 'x y z ?')
    return out



def lehmer_merge(clip, radius=3, passes=2):
    from vsutil import EXPR_VARS

    blur = [core.std.BoxBlur(i, hradius=radius, vradius=radius, hpasses=passes, vpasses=passes) for i in clip]

    count = len(clip)
    iterations = range(count)
    clipvars = EXPR_VARS[:count]
    blurvars = EXPR_VARS[count:]
    expr = ""

    for i in iterations:
        expr += f"{clipvars[i]} {blurvars[i]} - D{i}! "

    for i in iterations:
        expr += f"D{i}@ 3 pow "
    expr += "+ " * (count - 1)
    expr += "P1! "

    for i in iterations:
        expr += f"D{i}@ 2 pow "
    expr += "+ " * (count - 1)
    expr += "P2! "

    expr += "P2@ 0 = 0 P1@ P2@ / ? "

    for i in iterations:
        expr += f"{blurvars[i]} "
    expr += "+ " * (count - 1)
    expr += f"{count} / +"

    return core.akarin.Expr(clip + blur, expr)



def basedAA(clip, height, gamma):
    double = nnedi3cl_double(clip).resize.Bicubic(format = vs.RGBS)
    down = SSIM_downsample(double, h = height, w = get_w(height), filter_param_a = -1/2, filter_param_b = 1/4).resize.Bicubic(format = vs.GRAY16, matrix = 1)
    eedi = eedi3s(down.std.Transpose(), gamma = gamma)
    eedi = eedi3s(eedi.std.Transpose(), gamma = gamma).resize.Bicubic(format = vs.RGBS)
    out = SSIM_downsample(eedi, h = 1080, w = 1920, filter_param_a = -1/2, filter_param_b = 1/4).resize.Bicubic(format = vs.GRAY16, matrix = 1).resize.Bicubic(format = vs.GRAY16, matrix = 1)
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



def ccdmod(src: vs.VideoNode, threshold: float = 4, matrix: int = 1) -> vs.VideoNode:
    """
    A faster implementation that processes only chroma. Taken from jvsfunc.
    """

    if src.format.color_family != vs.YUV:
        raise ValueError('ccdmod: only YUV format is supported, use jvsfunc.ccd() for RGB.')

    if matrix is None:
        matrix = 1 if src.width > 1270 or src.height > 710 else 6

    thr = threshold**2/195075.0
    u = plane(src, 1)

    yuv = src.resize.Bicubic(u.width, u.height, format=vs.YUV444P16)
    rgbs = yuv.resize.Point(format=vs.RGBS, matrix_in=matrix)

    rrr = core.std.ShufflePlanes([rgbs, rgbs, rgbs], [0, 0, 0], vs.RGB)
    ggg = core.std.ShufflePlanes([rgbs, rgbs, rgbs], [1, 1, 1], vs.RGB)
    bbb = core.std.ShufflePlanes([rgbs, rgbs, rgbs], [2, 2, 2], vs.RGB)

    ex_ccd = core.akarin.Expr([yuv, rrr, ggg, bbb],
                              ['',
                               'y[-12,-12] y - 2 pow z[-12,-12] z - 2 pow a[-12,-12] a - 2 pow + + A! '
                               'y[-4,-12] y - 2 pow z[-4,-12] z - 2 pow a[-4,-12] a - 2 pow + + B! '
                               'y[4,-12] y - 2 pow z[4,-12] z - 2 pow a[4,-12] a - 2 pow + + C! '
                               'y[12,-12] y - 2 pow z[12,-12] z - 2 pow a[12,-12] a - 2 pow + + D! '
                               'y[-12,-4] y - 2 pow z[-12,-4] z - 2 pow a[-12,-4] a - 2 pow + + E! '
                               'y[-4,-4] y - 2 pow z[-4,-4] z - 2 pow a[-4,-4] a - 2 pow + + F! '
                               'y[4,-4] y - 2 pow z[4,-4] z - 2 pow a[4,-4] a - 2 pow + + G! '
                               'y[12,-4] y - 2 pow z[12,-4] z - 2 pow a[12,-4] a - 2 pow + + H! '
                               'y[-12,4] y - 2 pow z[-12,4] z - 2 pow a[-12,4] a - 2 pow + + I! '
                               'y[-4,4] y - 2 pow z[-4,4] z - 2 pow a[-4,4] a - 2 pow + + J! '
                               'y[4,4] y - 2 pow z[4,4] z - 2 pow a[4,4] a - 2 pow + + K! '
                               'y[12,4] y - 2 pow z[12,4] z - 2 pow a[12,4] a - 2 pow + + L! '
                               'y[-12,12] y - 2 pow z[-12,12] z - 2 pow a[-12,12] a - 2 pow + + M! '
                               'y[-4,12] y - 2 pow z[-4,12] z - 2 pow a[-4,12] a - 2 pow + + N! '
                               'y[4,12] y - 2 pow z[4,12] z - 2 pow a[4,12] a - 2 pow + + O! '
                               'y[12,12] y - 2 pow z[12,12] z - 2 pow a[12,12] a - 2 pow + + P! '
                               f'A@ {thr} < 1 0 ? B@ {thr} < 1 0 ? C@ {thr} < 1 0 ? D@ {thr} < 1 0 ? '
                               f'E@ {thr} < 1 0 ? F@ {thr} < 1 0 ? G@ {thr} < 1 0 ? H@ {thr} < 1 0 ? '
                               f'I@ {thr} < 1 0 ? J@ {thr} < 1 0 ? K@ {thr} < 1 0 ? L@ {thr} < 1 0 ? '
                               f'M@ {thr} < 1 0 ? N@ {thr} < 1 0 ? O@ {thr} < 1 0 ? P@ {thr} < 1 0 ? '
                               '+ + + + + + + + + + + + + + + 1 + Q! '
                               f'A@ {thr} < x[-12,-12] 0 ? B@ {thr} < x[-4,-12] 0 ? '
                               f'C@ {thr} < x[4,-12] 0 ? D@ {thr} < x[12,-12] 0 ? '
                               f'E@ {thr} < x[-12,-4] 0 ? F@ {thr} < x[-4,-4] 0 ? '
                               f'G@ {thr} < x[4,-4] 0 ? H@ {thr} < x[12,-4] 0 ? '
                               f'I@ {thr} < x[-12,4] 0 ? J@ {thr} < x[-4,4] 0 ? '
                               f'K@ {thr} < x[4,4] 0 ? L@ {thr} < x[12,4] 0 ? '
                               f'M@ {thr} < x[-12,12] 0 ? N@ {thr} < x[-4,12] 0 ? '
                               f'O@ {thr} < x[4,12] 0 ? P@ {thr} < x[12,12] 0 ? '
                               '+ + + + + + + + + + + + + + + x + Q@ /'])

    ex_ccd = ex_ccd.resize.Bicubic(src_left=-0.25)
    ex_ccd = depth(ex_ccd, get_depth(src))
    return core.std.ShufflePlanes([src, ex_ccd], [0, 1, 2], vs.YUV)







###SCENEFILTERING
opstart = 2882 #first frame game
opend = 5034
edstart = 29551 #first frame star
edend = 31695


op_lehmer1_ranges = [
    (0, 275), (1311, 1326), (1837, 1889),
]


op_bili_ranges = [
    (285, 554), (564, 715), (796, 1277),
    (1304, 1310), (1338, 1382), (1393, 1398),
    (1420, 1630), (1806, 1808)
]


op_bili_lines_ranges = [
    (1631, 1694), (1809, 1836), (1890, 1923)
]


ed_bili_ranges = [
    (0, 100), (179, 1013), (1308, 1323),
    (1353, 2040)
]


ed_bili_lines_ranges = [
    (1014, 1093)
]


bili_ranges = [
    (1666, 1669), (9660, 9698), (10418, 10445),
    (10448, 10489), (11521, 11567), (13886, 13916),
    (14088, 14347), (14451, 14538), (16198, 16244),
    (19473, 19683), (24969, 25023), (27117, 27209),
    (27782, 27816), (28563, 28824), (29167, 29241), 
]


bili_lines_ranges = [
    (515, 580), (8864, 8904), (9501, 9659),
    (9699, 9736), (11423, 11520), (13917, 14087),
    (24225, 24375), (25050, 25092), (27210, 27227),
    (27941, 28092), (32620, 32674), 
]


no_aa_ranges = [
    (opstart, opstart+275),
    (opstart+1288, opstart+1294),
    (opstart+1311, opstart+1353),
    (opstart+1388, opstart+1393),
    (opstart+1837, opstart+1889),
]


medium_aa_ranges = [
    (opstart+285, opstart+554),
    (opstart+716, opstart+732),
    (opstart+1924, opstart+2025),
    (23573, 23634), (23719, 23920), 
]


strong_aa_ranges = [
    (18149, 18241), 
]



###SOURCES
cbr = core.lsmas.LWLibavSource(r"C:\Users\twint\Downloads\Urusei Yatsura (2022) - 02 (Amazon Prime CBR 1080p).mkv")
vbr = core.lsmas.LWLibavSource(r"C:\Users\twint\Downloads\Urusei Yatsura (2022) - 02 (Amazon Prime VBR 1080p).mkv")
disney = core.lsmas.LWLibavSource(r"E:\Anime\Urusei Yatsura (2022) - S01E02 - 1080p WEB H.264 -NanDesuKa (DSNP).mkv")[24:]
bili = core.lsmas.LWLibavSource(r"C:\Users\twint\Downloads\Urusei Yatsura (2022) - 02 (B-Global 2160p).mkv").resize.Bicubic(format = vs.RGBS)
bili = SSIM_downsample(bili, w = 1920, h = 1080, filter_param_a = -1/2, filter_param_b = 1/4).resize.Bicubic(format = vs.YUV420P8, matrix = 1)



ncop = core.lsmas.LWLibavSource(r"C:\Users\twint\Desktop\uy_ncop.mp4")[6:opend-opstart+7]
nced = core.lsmas.LWLibavSource(r"C:\Users\twint\Desktop\uy_nced.mp4")[14:edend-edstart+15]

test = lehmer_merge([cbr, vbr, disney, bili], 3, 2)
test2 = lehmer_merge([cbr, vbr, disney])

bili_lines = mask.FreyChen().get_mask(get_y(bili)).std.Binarize(40).std.Maximum().std.Convolution([1]*9)
lehmbili = core.std.MaskedMerge(test, bili, bili_lines)


###OP SPLICING
lehmer1_op = test2[opstart:opend+1]
lehmer2_op = test[opstart:opend+1]
bili_op = bili[opstart:opend+1]

bili_lines_op = mask.FreyChen().get_mask(get_y(bili_op)).std.Binarize(40).std.Maximum().std.Convolution([1]*9)
lehmbili_op = core.std.MaskedMerge(lehmer2_op, bili_op, bili_lines_op)

splice_op = lvf.rfs(lehmer2_op, lehmer1_op, op_lehmer1_ranges)
splice_op = lvf.rfs(splice_op, bili_op, op_bili_ranges)
splice_op = lvf.rfs(splice_op, lehmbili_op, op_bili_lines_ranges)


###ED SPLICING
lehmer2_ed = test[edstart:edend+1]
bili_ed = bili[edstart:edend+1]

bili_lines_ed = mask.FreyChen().get_mask(get_y(bili_ed)).std.Binarize(40).std.Maximum().std.Convolution([1]*9)
lehmbili_ed = core.std.MaskedMerge(lehmer2_ed, bili_ed, bili_lines_ed)

splice_ed = lvf.rfs(lehmer2_ed, lehmbili_ed, ed_bili_lines_ranges)
splice_ed = lvf.rfs(splice_ed, bili_ed, ed_bili_ranges)



###AA
episode = test[:opstart] + splice_op + test[opend+1:edstart] + splice_ed + test[edend+1:]
episode = lvf.rfs(episode, bili, bili_ranges)
episode = lvf.rfs(episode, lehmbili, bili_lines_ranges)
episode = depth(episode, 16)


ep = get_y(episode)
aamask = ep.std.Prewitt().std.Binarize(50 << 8).std.Maximum().std.Convolution([1]*9)

aastrong = lvf.sraa(ep)
aastrong = core.std.MaskedMerge(ep, aastrong, aamask)

aamedium = basedAA(ep, 1620, 70)
aamedium = core.std.MaskedMerge(ep, aamedium, aamask)

nnedi = lvf.aa.taa(ep, lvf.aa.nnedi3(opencl=True))
aaweak = lvf.aa.clamp_aa(ep, nnedi, aastrong, 0.8)
aaweak = core.std.MaskedMerge(ep, aaweak, aamask)

aa = lvf.rfs(aaweak, ep, no_aa_ranges)
aa = lvf.rfs(aa, aamedium, medium_aa_ranges)
aa = lvf.rfs(aa, aastrong, strong_aa_ranges)

credits_op = vdf.dcm(bili, splice_op, ncop, prefilter = True, expand = 3, thr = 60, start_frame = opstart)
credits_ed = vdf.dcm(bili, splice_ed, nced, prefilter = True, expand = 3, thr = 60, start_frame = edstart)

credits = core.std.Expr([credits_op, credits_ed], "x y max")
credits = depth(credits_op, 16)

aa = core.std.MaskedMerge(aa, ep, credits)


###DEHALO
double = aa.nnedi3cl.NNEDI3CL(0, 1, 0, 0, 4, 3).std.Transpose().nnedi3cl.NNEDI3CL(0, 1, 0, 0, 4, 3).std.Transpose()
dehalo = FineDehalo(double, rx = 2.3, darkstr = 0, brightstr = 1).resize.Point(1920, 1080)
dehalo = core.std.ShufflePlanes([dehalo, episode], [0, 1, 2], vs.YUV)


###DERING
dering = EdgeCleaner(dehalo, strength = 5, rep = True, rmode = 17, smode = 1, hot = True)
deringmask = aamask.std.Inflate()
deringmasked = core.std.MaskedMerge(dehalo, dering, deringmask)


###DENOISE LUMA
deringluma = get_y(deringmasked)
deringluma32 = depth(deringluma, 32)

preden = SMDegrain(deringluma, tr = 2, thSAD = 220, RefineMotion = True, prefilter = 2)
preden32 = depth(preden, 32)

denoise_y = core.bm3dcuda.BM3D(clip = deringluma32, ref = preden32, sigma = 1.60, radius = 2).bm3d.VAggregate(radius = 2)
denoise_y = core.bm3dcuda.BM3D(clip = deringluma32, ref = depth(denoise_y, 32), sigma = 1.60, radius = 2).bm3d.VAggregate(radius = 2)

preden2 = SMDegrain(deringluma, tr = 2, thSAD = 150, RefineMotion = True, prefilter = 2)
preden232 = depth(preden2, 32)

denoise_y2 = core.bm3dcuda.BM3D(clip = deringluma32, ref = preden232, sigma = 1.00, radius = 2).bm3d.VAggregate(radius = 2)
denoise_y2 = core.bm3dcuda.BM3D(clip = deringluma32, ref = depth(denoise_y2, 32), sigma = 1.00, radius = 2).bm3d.VAggregate(radius = 2)

denoiseluma = lvf.rfs(denoise_y, denoise_y2, bili_ranges)
denoiseluma = lvf.rfs(denoiseluma, denoise_y2, (opstart, opend))
denoiseluma = lvf.rfs(denoiseluma, deringluma, op_lehmer1_ranges)


###DENOISE CHROMA
luma_merge = core.std.ShufflePlanes([denoise_y, deringmasked], [0, 1, 2], vs.YUV)
chromadn = ccdmod(luma_merge, threshold = 6.9)


###DEBAND
tmask = texture_mask(preden, 1)
tmask2 = texture_mask(preden, 2)
tmask3 = texture_mask(preden, 3)

linemask = retinex_edgemask(preden, 1)
debandmask = core.std.Expr([tmask, linemask], "x y +")
mediumdeband = dbs.placebo_deband(chromadn, 16, 3.0, 2, 0).std.MaskedMerge(chromadn, debandmask)


###GRAINING
grain = sizedgrn(mediumdeband, strength = 0.64, cstrength = 0, size = 1.15, sharp = 90, static = False, grain_chroma = False)
out = depth(grain, 10)


###OUTPUT
#out.set_output()


#out.set_output()
cbr.set_output(1)
bili.set_output(2)
#episode.set_output(2)
#bili.set_output(3)
#lehmbili.set_output(4)
#core.std.ShufflePlanes([aaweak, episode], [0, 1, 2], vs.YUV).set_output(5)
#core.std.ShufflePlanes([aamedium, episode], [0, 1, 2], vs.YUV).set_output(6)
#core.std.ShufflePlanes([aastrong, episode], [0, 1, 2], vs.YUV).set_output(7)
#out.set_output(8)




#scenefilter bilisplicing, aa, deband
#replace sraa