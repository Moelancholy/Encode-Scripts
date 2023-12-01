from adptvgrnMod import adptvgrnMod
from functools import partial
from muvsfunc import SSIM_downsample
from typing import List
from vsutil import *

import debandshit
import awsmfunc as awf
import ccd
import havsfunc as haf
import kagefunc as kgf
import lvsfunc as lvf
import vardefunc as vdf
from vsutil.clips import get_y
from vsutil.func import iterate
import xvs
import zzfunc as zzf

import vapoursynth as vs
core = vs.core



def detint(clip: vs.VideoNode) -> vs.VideoNode:
    clip = split(clip)
    y = clip[0].std.Expr("x 4096 <= x x 128 + ?")
    u = core.std.Expr([clip[1], clip[2]], "x 32768 = y 32768 = and x x 128 + ?")
    v = core.std.Expr([clip[1], clip[2]], "x 32768 = y 32768 = and y y 128 + ?")
    return join([y, u, v])


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
    rmask = vdf.mask.MinMax(range).get_mask(clip, lthr=0, multi=1.00)
    emask = ed_gray.std.Prewitt()
    em_hi = emask.std.Binarize(60 * 257, v0=65535, v1=0)
    em_hi = iterate(em_hi, core.std.Minimum, 5)
    em_me = emask.std.Binarize(40 * 257, v0=65535, v1=0)
    em_me = iterate(em_me, core.std.Minimum, 4)
    em_lo = emask.std.Binarize(20 * 257, v0=65535, v1=0)
    em_lo = iterate(em_lo, core.std.Minimum, 2)
    rm_txt = core.std.Expr([rmask, em_hi, em_me, em_lo], 'x y z a min min min')
    weighted = LevelsM(rm_txt, points=[x *256 for x in (1.75, 2.5, 5, 10)], levels=[0, 1, 1, 0], xpass=[0,0], return_expr=0)

    mask = weighted.std.BoxBlur(hradius=8,vradius=8).std.Expr(f'x {65535 * 0.2} - {1 / (1 - 0.2)} *')
    return mask



###SPLICE 1
opstart = 2758
opend = 4910
ncend = opend - opstart + 1



###SOURCES
src = core.lsmas.LWLibavSource(r"E:\Anime\[BDMV][アニメ] けいおん！\K_ON_BD_01\BDMV\STREAM\00000.m2ts")
src16 = depth(src, 16)
srcop = src[opstart:opend+1]
detintop = detint(depth(srcop, 16))
#srcy = srcy[7832] + srcy[3984] + srcy[4272] + srcy[5025] + srcy[6230] + srcy[6613] + srcy[7489] + srcy[9735] + srcy[15403] + srcy[17279] + srcy[24809] + srcy[27479] + srcy[28443] + srcy[28793] + srcy[32504]

comp = core.lsmas.LWLibavSource(r"E:\Anime\[Crow] K-ON! 1080P [BDRip 1920x1080 x264 FLAC Dual Audio]\[Crow] K-ON! 01 (BD x264 1920x1080 FLAC).mkv")
compy = get_y(comp)
#compy = compy[7832] + compy[3984] +compy[4272] + compy[5025] + compy[6230] + compy[6613] + compy[7489] + compy[9735] + compy[15403] + compy[17279] + compy[24809] + compy[27479] + compy[28443] + compy[28793] + compy[32504]

NCOP = core.lsmas.LWLibavSource(r"E:\Anime\[BDMV][アニメ] けいおん！\K_ON_BD_04\BDMV\STREAM\00003.m2ts")[:ncend]
NCED = core.lsmas.LWLibavSource(r"E:\Anime\[BDMV][アニメ] けいおん！\K_ON_BD_07\BDMV\STREAM\00004.m2ts")

shader = r"C:\Users\twint\AppData\Roaming\mpv\shaders\FSRCNNX_x2_56-16-4-1.glsl"



###SPLICE 2
OP_mask = vdf.dcm(src, credit_clip=srcop, nc_clip=NCOP, start_frame=opstart, thr=35, prefilter=True, expand=3)[opstart:opend+1]
OP_mask = OP_mask.resize.Bicubic(640, 360)
OP_mask = iterate(OP_mask, core.std.Maximum, 10).rgvs.RemoveGrain(20).rgvs.RemoveGrain(20).rgvs.RemoveGrain(20).rgvs.RemoveGrain(20).rgvs.RemoveGrain(20)
OP_mask = core.resize.Bicubic(OP_mask, 1920, 1080)
OP_mask = depth(OP_mask, 16)

NCOP = depth(NCOP, 16)
srcop = depth(srcop, 16)
detintop = depth(detintop, 16)
srcop = lvf.rfs(srcop, detintop, [(262, 329)])
maskedOP = core.std.MaskedMerge(NCOP, srcop, OP_mask)
maskedOP = lvf.rfs(maskedOP,srcop, [1451, 1516])

spliced = src16[:opstart] + maskedOP + src16[opend+1:]
srcy = get_y(spliced)

###TEXTUREMASK
ref = haf.SMDegrain(srcy, tr=3, prefilter=3, search=5, RefineMotion=True, thSAD=200, blksize = 8)
t_mask1 = texture_mask(ref, 3)



###AA
aa_mask = vdf.mask.SobelStd().get_mask(srcy, multi=1.50).std.Maximum().std.Minimum()
upscale = vdf.scale.fsrcnnx_upscale(srcy, width = 3840, height = 2160, downscaler = None, shader_file=shader, profile = 'zastin')
upscale2 = vdf.scale.nnedi3cl_double(srcy)
upscale_merge = core.std.Merge(upscale, upscale2, 0.4)

aascale1 = depth(SSIM_downsample(upscale_merge, w = 3200, h = 1800, sigmoid = True), 16)
aa1 = lvf.aa._eedi3_singlerate2(aascale1).std.Transpose() #edited singlerater, gamma=200
aa1 = lvf.aa._eedi3_singlerate2(aa1).std.Transpose()
aa1 = depth(SSIM_downsample(aa1, w = 1920, h = 1080, sigmoid = True), 16)
aa1 = core.std.MaskedMerge(srcy, aa1, aa_mask)



###DERING
aamerge = core.std.ShufflePlanes([aa1, spliced], [0, 1, 2], vs.YUV)
dering = haf.EdgeCleaner(aamerge, strength = 6, rep = True, rmode = 17, smode = 1, hot = True)



###CHROMA
c_rest = xvs.mwcfix(dering, kernel = 1, restore = 5, a = 2, grad = 2, warp = 6, thresh = 96, blur = 3, repair = 1)
line_m1 = vdf.mask.SobelStd().get_mask(srcy, multi=2.00).std.Maximum().std.Minimum().std.Binarize(84<<7)
c_dn_mask = core.std.Expr([line_m1, t_mask1], 'x y +')
c_denoise = ccd.ccd(c_rest, 5)
c_denoise = core.std.MaskedMerge(c_denoise, c_rest, c_dn_mask)



###DEBAND
light_deband1 = debandshit.dumb3kdb(c_denoise, radius = 16, threshold = [24, 24], grain = 24)
light_deband2 = debandshit.dumb3kdb(c_denoise, radius = 24, threshold = [40, 40], grain = 24)

medium_deband1 = debandshit.placebo_deband(c_denoise, radius = 16, threshold = [3.5, 3], grain = [5, 0])
medium_deband2 = debandshit.placebo_deband(c_denoise, radius = 20, threshold = [4, 3.5], grain = [6, 0])
medium_deband3 = debandshit.placebo_deband(c_denoise, radius = 24, threshold = [5, 4.5], grain = [8, 0])
medium_deband4 = debandshit.placebo_deband(c_denoise, radius = 12, threshold = [2.5, 2], grain = [5, 0])

average_deband = core.average.Mean([medium_deband1, medium_deband2, medium_deband3, medium_deband4])
medium_deband = core.std.MaskedMerge(average_deband, c_denoise, c_dn_mask)

mosaic_ass_looking_banding_deband = debandshit.placebo_deband(c_denoise, radius=40, threshold=[7, 12], grain=4)



###GRAIN
grain = adptvgrnMod(medium_deband, strength=0.06, size=1.30, luma_scaling=4, sharp=65, static=True, grain_chroma = False)
grain = adptvgrnMod(grain, strength=0.10, size=1.30, luma_scaling=7, sharp=65, static=False, grain_chroma = False)
out = depth(grain, 10)


src32 = depth(src, 32)
borders = core.std.AddBorders(get_y(src32), 8, 6, 4, 4).resize.Bicubic(src_left = 1)
descale_frac = core.descale.Debicubic(borders, width = 1280, height = 720, b=1/3, c=1/3)
rescale_frac = core.resize.Bicubic(descale_frac, 1934, 1088, filter_param_a = 1/3, filter_param_b = 1/3).resize.Bicubic(src_left=1).std.CropRel(6, 8, 4, 4)
rescale_720 = core.descale.Debicubic(get_y(src32), height = 720, width = 1280, b= 1/3, c = 1/3).resize.Bicubic(1920, 1080, filter_param_a=1/3, filter_param_b=1/3)

descale_test = core.descale.Debicubic

#rescale_frac = core.descale.Debicubic(get_y(src32), height = 720, src_top = 4, width = 1280, src_left = 7).resize.Bicubic(1920, 1080)


###OUTPUT
#awf.ScreenGen(clip = aa5, folder="Comparison", frame_numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], start=1, prefix ="xaa")
get_y(src).set_output(0)
rescale_frac.set_output(2)
rescale_720.set_output(3)
#maskedOP.set_output(4)
#grain.set_output(5)
#comp.set_output(6)


###TODO
# check chroma banding
# deband normal, strong, very strong
# normal and strong layered maybe with gradient mask, then texturemasked and linemasked
# very strong deband texturescaled into regular deband, maybe detailmask, also linemask, check gradientmask
# add grain (static bright, dynamic dark)
# lazy test encode
# quick comparison
# if found okay, scenefilter OP/ED
# supercut, test encode crf 17, use bitrate
# test encodes x265 cutree, no-cutree, x264 mbtree, no-mbtree (with/without qpfile) (qcomp, aq, psyrd, psyrdoq, cbqpoffs, cbrpoffset, ctu, max tu size)
# take best, and run test to tune crf






#7640, 24520