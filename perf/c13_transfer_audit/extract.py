#!/usr/bin/env python3
"""Pull every PREDICTED:/MEASURED: block out of the campaign state docs of record."""
import re, sys, glob, os
D="/home/moritz/.coworker/state"
pats=["b2z-*.md","b2z2-*.md","roof-*.md","ttx-*.md","c10-*.md","c12-*.md"]
files=[]
for p in pats: files+=sorted(glob.glob(os.path.join(D,p)))
KEY=re.compile(r"^(PREDICTED|PRE-REGISTERED|PREDICTION|MEASURED|VERDICT|DEFICIT-SECONDS|FOLD-SECONDS|SHIPPED)\s*:",re.I)
for f in files:
    lines=open(f,errors="replace").read().split("\n")
    out=[];i=0
    while i<len(lines):
        if KEY.match(lines[i]):
            blk=[(i+1,lines[i])];j=i+1
            while j<len(lines) and (lines[j].startswith(("  ","\t","* ","| ")) or lines[j].strip()=="" and False):
                blk.append((j+1,lines[j]));j+=1
            out.append(blk);i=j
        else: i+=1
    if out:
        print(f"\n########## {os.path.basename(f)}")
        for blk in out:
            for ln,tx in blk: print(f"{ln}: {tx}")
