#!/usr/bin/env bash
# tally.sh — per-model n / min / median / max, computed from draws.tsv (which index.sh derives
# from the logs). Every median quoted in a note or the state doc comes from here, never by hand.
awk -F'\t' '
  { split($2,a,"="); m=a[2]; split($5,b,"="); v=b[2]+0; if (b[2]=="NA") next; n[m]++; x[m,n[m]]=v }
  END { for (m in n) { c=n[m];
      for(i=1;i<=c;i++) for(j=i+1;j<=c;j++) if (x[m,i]>x[m,j]) { t=x[m,i]; x[m,i]=x[m,j]; x[m,j]=t }
      med = (c%2) ? x[m,(c+1)/2] : (x[m,c/2]+x[m,c/2+1])/2
      printf "%-16s n=%-2d min=%-12.6f med=%-12.6f max=%-12.6f band=%.2f%%\n",
             m, c, x[m,1], med, x[m,c], (x[m,c]/x[m,1]-1)*100 } }
' /home/ttuser/.coworker/wt/qb2-card-layer-baseline-reseed/perf/qb2cardlayer/draws.tsv | sort
