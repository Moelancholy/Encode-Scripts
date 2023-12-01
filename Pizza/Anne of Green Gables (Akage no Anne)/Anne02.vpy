import vapoursynth as vs
from vsutil import *

core = vs.core

###
title_card = 2035
credits = 35280

###GETTING THE SOURCES
src = core.lsmas.LWLibavSource(r"C:\Users\twint\Desktop\Encode\[BDMV][140326] Anne of Green Gables\ANNE_OF_GREEN_GABLES_1\BDMV\STREAM\00004.m2ts")[:37001]
ncop = core.lsmas.LWLibavSource(r"C:\Users\twint\Desktop\Encode\[BDMV][140326] Anne of Green Gables\ANNE_OF_GREEN_GABLES_7\BDMV\STREAM\00014.m2ts")
nced = core.lsmas.LWLibavSource(r"C:\Users\twint\Desktop\Encode\[BDMV][140326] Anne of Green Gables\ANNE_OF_GREEN_GABLES_7\BDMV\STREAM\00015.m2ts")[:1721]


###SPLICING
op_splice = ncop[:title_card] + src[title_card:]
ed_splice = op_splice[:credits] + nced
crop = core.std.CropRel(ed_splice, 242, 242, 0, 0)
crop = depth(crop, 16)


###REMAPPING
#remap = core.remap.RemapFrames(baseclip=crop, sourceclip=crop, filename=r"C:\Users\twint\Documents\GitHub\Encode-Scripts\Pizza\Anne of Green Gables (Akage no Anne)\Anne01.txt")

###OUTPUT
final = depth(crop, 10)
final.set_output()