#!/usr/bin/python
import sys,os
sys.path.append('/home/kuo/climdb')
os.environ.setdefault("DJANGO_SETTINGS_MODULE","climdb.settings")
from django.contrib.gis.geos import Point
import django
django.setup()

def importTIDECONS():
    import sys
    from scipy.io import loadmat
    from climsea.models import Gridxyid,Tideid,Tidecons
    
    f=loadmat('/home/kuo/data/clim/tideHUVcon_dx30_105E135E_2N35N_7TPXO.mat')
    x=f['lon']
    y=f['lat']
    con=f['con']
    ph=f['ph']
    ha=f['ha']
    hp=f['hp']
    ua=f['ua']
    up=f['up']
    va=f['va']
    vp=f['vp']
 
    for i in range(len(x)):
       for j in range(len(x[0])):
          aobj=Gridxyid.objects.get(longitude=round(x[i][j],6),latitude=round(y[i][j],6),dxy=30)
          if aobj.waterio:
             for k in range(len(ha[0][0])):
                bobj=Tideid.objects.get(tidename=con[k])
                Tidecons.objects.update_or_create(gridxyid=aobj,tideid=bobj,defaults={\
                         'ha':round(ha[i][j][k],6),'hp':round(hp[i][j][k],6),\
                         'ua':round(ua[i][j][k],6),'up':round(up[i][j][k],6),\
                         'va':round(va[i][j][k],6),'vp':round(vp[i][j][k],6)})
       print i
    return

def main():
    importTIDECONS()

if __name__ == '__main__':
    main()

