from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from .models import Gridxyid

def gettidexyjson(request,x,y):
    from django.core import serializers
    from .models import Tideid,Tidecons
    from django.contrib.gis.geos import Point,Polygon
    m=0.05
    x,y=float(x),float(y)
    p0=Point((x,y))
    geom=Polygon.from_bbox((x-m,y-m,x+m,y+m))
    a=Gridxyid.objects.filter(coord__within=geom,waterio=True).distance(p0).order_by('distance')
    if a.count()<1:
        return HttpResponse('[{}]',content_type='application/json')
    a=a[0]
    b=Tidecons.objects.filter(gridxyid=a.id)
    w='['
    for i in range(len(b)):
        w=w+'{"freq":'+str(b[i].tideid.freq)+',"ph":'+str(b[i].tideid.ph)+\
            ',"ha":'+str(b[i].ha)+',"hp":'+str(b[i].hp)+\
            ',"ua":'+str(b[i].ua)+',"up":'+str(b[i].up)+\
            ',"va":'+str(b[i].va)+',"vp":'+str(b[i].vp)+'},'
    w=w[:-1]+']'
    #c=serializers.serialize('json',b,fields=('ha','hp','ua','up','va','vp'))
    return HttpResponse(w,content_type='application/json')

def getclimxyjson(request,vn,ymd,depth):
    from .models import Climtimid,Climdata
    tm=Climtimid.objects.filter(ymd=int(ymd))
    #dxy=4 if vn in 'ts' or int(ymd)>19000000 else 1
    dxy=4
    xy=Gridxyid.objects.filter(waterio=True,dxy=dxy)
    aa=Climdata.objects.filter(climtimid=tm,gridxyid__in=xy,z=int(depth)).order_by('gridxyid__longitude','gridxyid__latitude')
    gg=str(map(list,zip(list(aa.values_list('gridxyid__longitude',flat=True)),\
                        list(aa.values_list('gridxyid__latitude',flat=True)),\
                        list(aa.values_list(vn,flat=True)))))
    return HttpResponse(gg,content_type='application/json')

def getclimxycsv(request,vn,ymd,depth):
    from .models import Climtimid,Climdata
    tm=Climtimid.objects.filter(ymd=int(ymd))
    #dxy=4 if vn in 'ts' or int(ymd)>19000000 else 1
    dxy=4
    #xy=Gridxyid.objects.filter(waterio=True,dxy=dxy)
    xy=Gridxyid.objects.filter(dxy=dxy)
    aa=Climdata.objects.filter(climtimid=tm,gridxyid__in=xy,z=int(depth)).order_by('gridxyid__longitude','gridxyid__latitude')
    x=list(aa.values_list('gridxyid__longitude',flat=True))
    y=list(aa.values_list('gridxyid__latitude',flat=True))
    z=list(aa.values_list(vn,flat=True))
    gg="%LongE,%LatN,%data\n"
    for i in range(len(x)):
        gg=gg+str(round(x[i],2))+','+str(round(y[i],2))+','+str(z[i])+'\n'
        #           ('NaN' if b[j][i]>1 else str(round(a[j][i],4)-273.1500))+'\n'
    kk=HttpResponse(gg,content_type='text/csv')
    kk['Content-Disposition']='attachment; filename=Hidy_xymap_'+vn+'_'+str(depth)+'m_'+str(ymd)+'.csv'
    return kk

def getclimxygeojson(request,vn,ymd,depth):
    from .models import Climtimid,Climdata
    from django.contrib.gis.geos import Point
    from django.core import serializers
    tm=Climtimid.objects.filter(ymd=int(ymd))
    #dxy=4 if vn in 'ts' or int(ymd)>19000000 else 1
    dxy=4
    xy=Gridxyid.objects.filter(waterio=True,dxy=dxy)
    aa=Climdata.objects.filter(climtimid=tm,gridxyid__in=xy,z=int(depth)).order_by('gridxyid__longitude','gridxyid__latitude')
    #gg=serializers.serialize('geojson',aa,geometry_field='xy',fields=(vn))
    a=list(aa.values_list(vn,flat=True))
    x=list(aa.values_list('gridxyid__longitude',flat=True))
    y=list(aa.values_list('gridxyid__latitude',flat=True))
    gg='{"type":"FeatureCollection","crs":{"type":"name","properties":{"name":"EPSG:4326"}},"features":['+\
        '{"type":"Feature","geometry":{"type":"Point","coordinates":['+str(x[0])+','+str(y[0])+']},'+\
        '"properties":{"weight":'+str(a[0])+'}}'
    for i in range(1,len(x)):
        gg=gg+',{"type":"Feature","geometry":{"type":"Point","coordinates":['+str(x[i])+','+str(y[i])+']},'+\
            '"properties":{"weight":'+str(a[i])+'}}'
    gg=gg+']}'
    return HttpResponse(gg,content_type='application/json')

def getclimxycontour(request,vn,ymd,depth):
    #from .models import Climtimid,Climdata
    from wsgiref.util import FileWrapper
    f='/home/kuo/figs/clim/'+vn+'/'+depth+'/'+ymd+'.geojson'
    return HttpResponse(FileWrapper(file(f)),content_type='application/json')

def getclimxycntbnd(request,vn,ymd,depth):
    #from .models import Climtimid,Climdata
    from wsgiref.util import FileWrapper
    f='/home/kuo/figs/clim/'+vn+'/'+depth+'/'+ymd+'.geojson.json'
    return HttpResponse(FileWrapper(file(f)),content_type='application/json')

def getclimzsect(request,vn,ymd,lonlat):
    from .models import Climtimid,Climdata
    from wsgiref.util import FileWrapper
    f='/home/kuo/figs/clim/'+vn+'/z/'+lonlat+'/'+ymd+'.json'
    return HttpResponse(FileWrapper(file(f)),content_type='application/json')

def getclimzsectcsv(request,vn,ymd,lonlat):
    from .models import Climtimid,Climdata
    tm=Climtimid.objects.filter(ymd=int(ymd))
    #dxy=4 if vn in 'ts' or int(ymd)>19000000 else 1
    dxy=4
    #xy=Gridxyid.objects.filter(waterio=True,dxy=dxy)
    lonlat=float(lonlat)/100.0
    if float(lonlat)>100.0:
      xy=Gridxyid.objects.filter(dxy=dxy,longitude=lonlat)
      ixy='gridxyid__latitude'
      jxy='E'
      gg="%LatN,%Depth,%data\n"
    else:
      xy=Gridxyid.objects.filter(dxy=dxy,latitude=lonlat)
      ixy='gridxyid__longitude'
      jxy='N'
      gg="%LongE,%Depth,%data\n"
    aa=Climdata.objects.filter(climtimid=tm,gridxyid__in=xy,z__lte=500).order_by('z',ixy)
    x=list(aa.values_list(ixy,flat=True))
    y=list(aa.values_list('z',flat=True))
    z=list(aa.values_list(vn,flat=True))
    for i in range(len(x)):
        gg=gg+str(round(x[i],2))+','+str(y[i])+','+str(z[i])+'\n'
        #           ('NaN' if b[j][i]>1 else str(round(a[j][i],4)-273.1500))+'\n'
    kk=HttpResponse(gg,content_type='text/csv')
    kk['Content-Disposition']='attachment; filename=Hidy_zsect_'+vn+'_'+str(int(lonlat*100.))+jxy+'_'+str(ymd)+'.csv'
    return kk

def getclimtimes(request,vn,x,y,depth):
    from .models import Climtimid,Climdata
    tm=Climtimid.objects.filter(yr__gt=1800)
    #dxy=4 if vn in 'ts' or int(ymd)>19000000 else 1
    dxy=4
    xy=Gridxyid.objects.filter(longitude=round(float(x),2),latitude=round(float(y),2),dxy=dxy)
    aa=Climdata.objects.filter(gridxyid=xy,climtimid__in=tm,z=int(depth)).order_by('climtimid__ymd')
    gg=str(map(list,zip(list(aa.values_list('climtimid__ymd',flat=True)),\
                        list(aa.values_list(vn,flat=True)))))
    return HttpResponse(gg,content_type='application/json')

def getclimprofz(request,vn,x,y,ymd):
    from .models import Climtimid,Climdata
    tm=Climtimid.objects.filter(ymd=int(ymd))
    #dxy=4 if vn in 'ts' or int(ymd)>19000000 else 1
    dxy=4
    xy=Gridxyid.objects.filter(longitude=round(float(x),2),latitude=round(float(y),2),dxy=dxy)
    aa=Climdata.objects.filter(gridxyid=xy,climtimid=tm).order_by('z')
    #gg=str(map(list,zip(list(aa.values_list('z',flat=True)),\
    #                    list(aa.values_list(vn,flat=True)))))
    gg='['+'{"z":'+str(list(aa.values_list('z',flat=True)))+\
           ',"d":'+str(list(aa.values_list(vn,flat=True)))+'}]'
    return HttpResponse(gg,content_type='application/json')

def getclimlzjson(request,vn,ymd,lonlat):
    from .models import Climtimid,Climdata
    tm=Climtimid.objects.filter(ymd=int(ymd))
    #dxy=4 if vn in 'ts' or int(ymd)>19000000 else 1
    dxy=4
    lonlat=float(round(float(lonlat)*float(dxy),0))/float(dxy)
    if float(lonlat)>100.0:
        xy=Gridxyid.objects.filter(longitude=lonlat,dxy=dxy)
        a='latitude'
    else:
        xy=Gridxyid.objects.filter(latitude=lonlat,dxy=dxy)
        a='longitude'
    aa=Climdata.objects.filter(climtimid=tm,gridxyid__in=xy)
    x=list(aa.values_list('gridxyid__'+a,flat=True).order_by('gridxyid__'+a).distinct())
    z=list(aa.order_by('z','gridxyid__'+a).values_list(vn,flat=True))
    gg='['+'{"x":'+str(x)+\
           ',"y":[0,-10,-20,-30,-50,-75,-100,-125,-150,-200,-250,-300,-400,-500]'+\
           ',"z":'+str(z)+'}]'
    return HttpResponse(gg,content_type='application/json')

def getsectzslow(request,vn,ymd,lonlat,dz):
    from .models import Climtimid,Climdata
    from scipy.interpolate import interp1d
    import numpy
    tm=Climtimid.objects.filter(ymd=int(ymd))
    #dxy=4 if vn in 'ts' or int(ymd)>19000000 else 1
    dxy=4
    dx=0.25
    lonlat=float(round(float(lonlat)*float(dxy),0))/float(dxy)
    if float(lonlat)>100.0:
        xy=Gridxyid.objects.filter(longitude=lonlat,dxy=dxy)
        a='latitude'
    else:
        xy=Gridxyid.objects.filter(latitude=lonlat,dxy=dxy)
        a='longitude'
    aa=Climdata.objects.filter(climtimid=tm,gridxyid__in=xy)
    x=list(xy.values_list(a,flat=True).order_by(a))
    y=list(aa.order_by('z').values_list('z',flat=True).distinct())
    #y=[0,10,20,30,50,75,100,125,150,200,250,300,400,500]
    dz=int(dz)
    nx,ny,nz=len(x),len(y),500/dz+1
    #yy=[i*dz for i in range(nz)]
    #yn=numpy.asarray(yy)
    yy=numpy.linspace(0,500,nz)
    ng=[-999 for i in range(nz)]
    #bb=list(aa.order_by('gridxyid__'+a,'z').values_list(vn,flat=True))
    bb=numpy.asarray(list(aa.order_by('gridxyid__'+a,'z').values_list(vn,flat=True)))
    gb=[]
    for i in range(nx):
      #g=numpy.asarray(bb[i*ny:(i+1)*ny])
      g=bb[i*ny:(i+1)*ny]
      n=sum(g>-900.0)
      if n>1:
        k=interp1d(y[:n],g[:n],fill_value=-999)
        #n=sum(yn<=y[n-1])
        n=numpy.argmax(yy>=y[n-1])
        #gb=gb+list(k(yy[:n]))
        k=list(numpy.round(k(yy[:n]),4))
        if n<nz:
          #gb=gb+[-999 for k in range(nz-n)]
          k=k+list(ng[n:])
      else:
        #gb=gb+ng
        k=ng
      gb=gb+k
    #gg='['+'{"x":'+str(x)+',"y":'+str(yy).replace(' ','-')+',"z":'+str(gb)+'}]'
    gg='['+'{"x":'+str(x)+',"y":'+str(list(yy*-1))+',"z":'+str(list(numpy.asarray(gb).reshape(nx,nz).T.flatten()))+'}]'
    return HttpResponse(gg,content_type='application/json')

def getclimsegtrackjson(request,vn,ymd,depth):
    from .models import Climtimid,Climdata
    from django.contrib.gis.geos import GEOSGeometry,Point,LineString,Polygon
    from geopy.distance import vincenty
    depth=int(depth)
    #idxy=4 if vn in 'ts' or int(ymd)>19000000 else 1
    idxy=4
    ibuf=1.0/float(idxy)
    tm=Climtimid.objects.filter(ymd=int(ymd))
    xy=request.GET.get('line',0)
    xy=GEOSGeometry('SRID=4326;LINESTRING '+xy)
    x,y,z,d='"x":[','"y":[','"z":[','"d":['
    dd=0
    for j in range(len(xy)-1):
      n=min(int(LineString((xy[j],xy[j+1])).length/ibuf),100)-1
      if (n>0):
        x1,y1,x2,y2=xy[j][0],xy[j][1],xy[j+1][0],xy[j+1][1]
        dx,dy=(x2-x1)/n,(y2-y1)/n
        for i in range(n):
          pp=Point((x1+i*dx,y1+i*dy),srid=4326)
          geom=Polygon.from_bbox((x1+i*dx-ibuf,y1+i*dy-ibuf,x1+i*dx+ibuf,y1+i*dy+ibuf))
          a=Gridxyid.objects.filter(dxy=idxy,coord__within=geom).distance(pp).order_by('distance')[0]
          x,y=x+str(round(x1+i*dx,2))+',',y+str(round(y1+i*dy,2))+','
          z=z+str(round(list(Climdata.objects.filter(gridxyid=a,climtimid=tm,z=depth).values_list(vn,flat=True))[0],3))+','
          d=d+str(int(dd+vincenty((y1+i*dy,x1+i*dx),(y1,x1)).kilometers))+','
        dd=dd+vincenty((y2,x2),(y1,x1)).kilometers
    pp=Point((x2,y2),srid=4326)
    geom=Polygon.from_bbox((x2-ibuf,y2-ibuf,x2+ibuf,y2+ibuf))
    a=Gridxyid.objects.filter(dxy=idxy,coord__within=geom).distance(pp).order_by('distance')[0]
    x,y=x+str(round(x2,2))+'],',y+str(round(y2,2))+'],'
    z=z+str(round(list(Climdata.objects.filter(gridxyid=a,climtimid=tm,z=depth).values_list(vn,flat=True))[0],3))+'],'
    d=d+str(int(dd))+']'
    a='[{'+x+y+z+d+'}]'
    return HttpResponse(a,content_type='application/json')

