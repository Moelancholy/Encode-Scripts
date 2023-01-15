import lvsfunc as lvf
import vsmasktools as mask

from adptvgrnmod import sizedgrn
from jvsfunc import ccdmod, retinex_edgemask
from havsfunc import SMDegrain, FineDehalo, EdgeCleaner
from vodesfunc import NNEDI_Doubler
from vsaa import upscaled_sraa, Eedi3SR
from vsdeband import Placebo
from vsdenoise import chroma_reconstruct
from vskernels import BicubicDidee
from vsrgtools import lehmer_diff_merge
from vsscale import ssim_downsample, Waifu2x
from vstools import replace_ranges, core, vs, plane
from vsutil import depth, get_y, iterate, get_w

core.max_cache_size = 40*1024
core.num_threads = 12


###CUSTOM FUNCS
def eedi3s(clip, field=0, alpha=0.125, beta=0.25, gamma=50, nrad=2, mdis=20, vcheck=2, vthresh0=12, vthresh1=24, vthresh2=4, mclip=None): 
    out = clip.eedi3m.EEDI3(field, False, 0, alpha, beta, gamma, nrad, mdis, vcheck=vcheck, vthresh0=vthresh0, vthresh1=vthresh1, vthresh2=vthresh2, sclip=clip, mclip=mclip)
    if mclip is not None:
        return core.std.Expr([mclip, out, clip], 'x y z ?')
    return out


def basedAA(clip, height, gamma):
    double = NNEDI_Doubler().double(clip)
    down = ssim_downsample(double, width = get_w(height), height = height, smooth = 1, scaler = BicubicDidee)
    eedi = eedi3s(down.std.Transpose(), gamma = gamma)
    eedi = eedi3s(eedi.std.Transpose(), gamma = gamma)
    out = ssim_downsample(double, width = 1920, height = 1080, smooth = 1, scaler = BicubicDidee)
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
    rmask = mask.MinMax(range).edgemask(clip, lthr=0, multi=1.00)
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


def clamp_aa(src: vs.VideoNode, weak: vs.VideoNode, strong: vs.VideoNode, strength: float = 1) -> vs.VideoNode:
    """
    Clamp stronger AAs to weaker AAs.
    Useful for clamping upscaled_sraa or eedi3 to nnedi3 for a strong but precise AA.
    Stolen from Zastin.
    :param src:      Non-AA'd source clip.
    :param weak:     Weakly-AA'd clip (eg: nnedi3)
    :param strong:   Strongly-AA'd clip (eg: eedi3)
    :param strength: Clamping strength (Default: 1)
    :return:         Clip with clamped anti-aliasing.
    """
    if src.format is None or weak.format is None or strong.format is None:
        raise ValueError("clamp_aa: 'Variable-format clips not supported'")
    thr = strength * (1 << (src.format.bits_per_sample - 8)) if src.format.sample_type == vs.INTEGER \
        else strength/219
    clamp = core.std.Expr([get_y(src), get_y(weak), get_y(strong)],
                          expr=f"x y - x z - xor x x y - abs x z - abs < z y {thr} + min y {thr} - max z ? ?"
                          if thr != 0 else "x y z min max y z max min")
    return clamp if src.format.color_family == vs.GRAY \
        else core.std.ShufflePlanes([clamp, src], planes=[0, 1, 2], colorfamily=vs.YUV)


###SCENEFILTERING
opstart = 1726 #first frame purple
opend = 3884
ed_ranges = [(29207, 31362)]

no_aa_ranges = [
    (18727, 19939), (28021, 28162), 
]


bili_ranges = [
    (opstart, opstart+58), (opstart+334, opstart+616),
    (opstart+698, opstart+728), (opstart+1173, opstart+1360),
    (opstart+1382, opstart+1472), (opstart+1504, opstart+1593),
    (opstart+1651, opstart+1681), (opstart+1824, opstart+1863),
    (opstart+1894, opstart+1931), (opstart+1968, opstart+2038),
    (4740, 4798), (4846, 4860), (7125, 7153), (10276, 10345),
    (10567, 10632), (12458, 12478), (13145, 13168),
    (13606, 13712), (13928, 13987), (26067, 26135),
    (27034, 27055), (27479, 27525), (28516, 28694),
    (28969, 29115), 
]


bili_lines_ranges = [
    (opstart+59, opstart+63), (opstart+192, opstart+288),
    (opstart+653, opstart+697), (opstart+729, opstart+1172),
    (opstart+1361, opstart+1381), (opstart+1473, opstart+1744),
    (opstart+1594, opstart+1650), (opstart+1682, opstart+1823),
    (opstart+1864, opstart+1893), (opstart+1932, opstart+1967),
    (opstart+2039, opstart+2091), (4799, 4845), (4861, 5092),
    (5128, 5139), (7154, 7188), (7278, 7324), (9369, 9447),
    (10190, 10275), (10633, 10673), (12852, 12988),
    (13060, 13103), (13193, 13274), (13808, 13836),
    (13988, 14013), (14749, 15063), (15117, 15164),
    (15208, 15300), (15330, 15361), (15379, 15418),
    (15788, 15906), (16259, 16279), (25418, 25453),
    (25500, 25647), (26799, 26956), (27243, 27288),
    (27526, 27560), (29116, 29206)
]


medium_aa_ranges = [
    (742, 872), (1197, 1553), (1642, 1725), (5134, 5200),
    (5328, 5446), (5899, 6137), (6579, 6629), (6823, 6894),
    (7854, 7914), (8110, 8168), (8204, 8289), (8331, 8414),
    (10464, 10538), (10674, 10730), (11587, 11772),
    (11890, 12226), (12296, 12365), (12479, 12851),
    (13427, 13605), (14451, 14748), (15165, 15207),
    (16497, 16525), (16728, 16809), (16983, 17077),
    (18484, 18587), (20933, 21431), (21586, 21608),
    (22689, 23130), (23805, 24022), (24214, 24916),
    (25744, 25894), (27855, 28020), (28319, 28515),
]


strong_aa_ranges = [
    (opstart+269, opstart+333), (opstart+459, opstart+467),
    (4993, 5092), (9111, 9217), (13330, 13426), 
    (16432, 16496), (17840, 17929), (20034, 20116),
]



###SOURCES
cbr = core.lsmas.LWLibavSource(r"C:\Users\twint\Downloads\Urusei Yatsura (2022) - 13 (Amazon Prime CBR 1080p).mkv")
vbr = core.lsmas.LWLibavSource(r"C:\Users\twint\Downloads\Urusei Yatsura (2022) - 13 (Amazon Prime VBR 1080p).mkv")
disney = core.lsmas.LWLibavSource(r"C:\Users\twint\Downloads\Urusei Yatsura (2022) - 13 (Disney+ 1080p).mkv")[24:-24]
bili = core.lsmas.LWLibavSource(r"C:\Users\twint\Downloads\Urusei Yatsura (2022) - 13 (B-Global 2160p).mkv")
bili = depth(ssim_downsample(depth(bili, 16), 1920, 1080, 1, BicubicDidee), 8)

test = lehmer_diff_merge([disney, vbr, cbr, bili])
test2 = lehmer_diff_merge([disney, vbr, cbr])

bili_lines = mask.FreyChen().edgemask(get_y(bili)).std.Binarize(40).std.Maximum().std.Convolution([1]*9)
lehmbili = core.std.MaskedMerge(test, bili, bili_lines)


###AA
episode = replace_ranges(test, lehmbili, bili_lines_ranges)
episode = replace_ranges(test, bili, bili_ranges)
episode = depth(episode, 16)


ep = get_y(episode)
aamask = ep.std.Prewitt().std.Binarize(50 << 8).std.Maximum().std.Convolution([1]*9)

aastrong = upscaled_sraa(ep, aafunc=Eedi3SR(hp=False))
aastrong = core.std.MaskedMerge(ep, aastrong, aamask)

aaweak = basedAA(ep, 1620, 50)
aaweak = core.std.MaskedMerge(ep, aaweak, aamask)

aamedium = upscaled_sraa(ep, rfactor = 2.0, aafunc=Eedi3SR(hp=False, alpha = 0.14, beta = 0.5, gamma = 400))
aamedium = core.std.MaskedMerge(ep, aamedium, aamask)

aa = replace_ranges(aaweak, ep, no_aa_ranges)
aa = replace_ranges(aa, aamedium, medium_aa_ranges)
aa = replace_ranges(aa, aastrong, strong_aa_ranges)


###DEHALO
double = aa.nnedi3cl.NNEDI3CL(0, 1, 0, 0, 4, 3).std.Transpose().nnedi3cl.NNEDI3CL(0, 1, 0, 0, 4, 3).std.Transpose()
dehalo = FineDehalo(double, rx = 2.3, darkstr = 0, brightstr = 1).resize.Point(1920, 1080)
dehalo = core.std.ShufflePlanes([dehalo, episode], [0, 1, 2], vs.YUV)


###DERING
dering = EdgeCleaner(dehalo, strength = 5, rep = True, rmode = 17, smode = 1, hot = True)
deringmask = aamask.std.Inflate()
deringmasked = core.std.MaskedMerge(dehalo, dering, deringmask)


###FIX ED
ed_down = core.resize.Bicubic(deringmasked, 960, 540, format=vs.RGBS)
ed_up = Waifu2x(cuda = True).scale(ed_down, 1920, 1080, tiles =2, overlap = 8).resize.Bicubic(format=vs.YUV444P16, matrix = 1)
#ed_up = core.std.ShufflePlanes([deringmasked, ed_up], [0,1,2], vs.YUV)

uplane = plane(ed_up, 1)
vplane = plane(ed_up, 2)

uaa = upscaled_sraa(uplane, rfactor = 1.5, aafunc=Eedi3SR(hp=False, alpha = 0.30, beta = 0.5, gamma = 60))
vaa = upscaled_sraa(vplane, rfactor = 1.5, aafunc=Eedi3SR(hp=False, alpha = 0.30, beta = 0.5, gamma = 60))
uvaa = core.std.ShufflePlanes([ed_up, uaa, vaa], [0,0,0], vs.YUV).resize.Bicubic(format=vs.YUV420P16, matrix = 1)
uvaa = core.std.ShufflePlanes([deringmasked, uvaa], [0,1,2], vs.YUV)

ed_fix = replace_ranges(deringmasked, uvaa, ed_ranges)


###DENOISE LUMA
deringluma = get_y(ed_fix)
deringluma32 = depth(deringluma, 32)

preden = SMDegrain(deringluma, tr = 2, thSAD = 220, RefineMotion = True, prefilter = 2)
preden32 = depth(preden, 32)

denoise_y = core.bm3dcuda.BM3D(clip = deringluma32, ref = preden32, sigma = 1.60, radius = 2).bm3d.VAggregate(radius = 2)
denoise_y = core.bm3dcuda.BM3D(clip = deringluma32, ref = depth(denoise_y, 32), sigma = 1.60, radius = 2).bm3d.VAggregate(radius = 2)

preden2 = SMDegrain(deringluma, tr = 2, thSAD = 150, RefineMotion = True, prefilter = 2)
preden232 = depth(preden2, 32)

denoise_y2 = core.bm3dcuda.BM3D(clip = deringluma32, ref = preden232, sigma = 1.00, radius = 2).bm3d.VAggregate(radius = 2)
denoise_y2 = core.bm3dcuda.BM3D(clip = deringluma32, ref = depth(denoise_y2, 32), sigma = 1.00, radius = 2).bm3d.VAggregate(radius = 2)

denoiseluma = replace_ranges(denoise_y, denoise_y2, bili_ranges)
denoiseluma = replace_ranges(denoiseluma, denoise_y2, ed_ranges)


###DENOISE CHROMA
luma_merge = core.std.ShufflePlanes([denoise_y, ed_fix], [0, 1, 2], vs.YUV)
chromadn = ccdmod(luma_merge, threshold = 6.9)


###DEBAND
tmask = texture_mask(preden, 1)
tmask2 = texture_mask(preden, 2)
tmask3 = texture_mask(preden, 3)

linemask = retinex_edgemask(preden, 1)
debandmask = core.std.Expr([tmask, linemask], "x y +")
mediumdeband = Placebo.deband(chromadn, 16, 3.0, 2, 0).std.MaskedMerge(chromadn, debandmask)

###GRAINING
grain = sizedgrn(mediumdeband, strength = [0.77, 0], size = 1.20, sharp = 90, static = False, temporal_average = 15)
out = depth(grain, 10)


###OUTPUT
#cbr.set_output(1)
#vbr.set_output(2)
#disney.set_output(3)
#bili.set_output(4)
#episode.set_output(5)



out.set_output()
#cbr.set_output(1)
#episode.set_output(2)
#bili.set_output(3)
#lehmbili.set_output(4)
#core.std.ShufflePlanes([aaweak, episode], [0, 1, 2], vs.YUV).set_output(5)
#core.std.ShufflePlanes([aamedium, episode], [0, 1, 2], vs.YUV).set_output(6)
#core.std.ShufflePlanes([aastrong, episode], [0, 1, 2], vs.YUV).set_output(7)
#out.set_output(8)



#