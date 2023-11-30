#!/usr/bin/env python
import numpy as np
import pandas as pd
from astropy.io import fits
import torch
import torch.nn.functional as F
import torch.nn as nn
import pickle
from astropy.table import Table
import argparse
import os,sys
pd.options.mode.chained_assignment = None
import pkg_resources


def makeparser():
    example_text = '''Examples:
    auriga test.fits --tableOut test-out --saveFlux test --tutorial \n
    auriga test.csv --localFlux --iters=20 --tutorial\n
    auriga test.fits --localFlux --gaiaFluxErrors --g phot_g_mean_mag --bp phot_bp_mean_mag --rp phot_rp_mean_mag --j j_m --h h_m --k ks_m --ej j_msigcom --eh h_msigcom --ek ks_msigcom --eparallax parallax_error --tutorial
    '''
    
    parser = argparse.ArgumentParser(description='Running Auriga neural net to determine age, extinction, and distance to a stellar population',epilog=example_text,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tableIn",help="Input table with Gaia DR2 source ids and cluster ids",default='',nargs='?')
    parser.add_argument("--tutorial",help="Use included test.fits or test.csv files as inputs",action='store_true')
    parser.add_argument("--tableOut",help="Prefix of the csv file into which the cluster properties should be written, default tableIn-out",default='')
    parser.add_argument("--iters", type=int,help="Number of iterations of each cluster is passed through Auriga to generate the errors, default 10",default=10)
    parser.add_argument("--localFlux",help="Download necessary flux from Gaia archive for all source ids, default True",action='store_true')
    parser.add_argument("--saveFlux",help="If downloading flux, prefix of file where to save it, default empty",default='')
    parser.add_argument("--silent",help="Suppress print statements, default False",action='store_true')
    parser.add_argument("--cluster",help="Column with cluster membership",default='cluster')
    parser.add_argument("--source_id",help="Column with Gaia DR2 source id,",default='source_id')
    parser.add_argument("--gaiaFluxErrors",help="If loading flux, whether uncertainties in Gaia bands have been converted from flux to magnitude, default True",action='store_true')
    parser.add_argument("--g",help="If loading flux, column for G magnitude",default='g')
    parser.add_argument("--bp",help="If loading flux, column for BP magnitude",default='bp')
    parser.add_argument("--rp",help="If loading flux, column for RP magnitude",default='rp')
    parser.add_argument("--j",help="If loading flux, column for J magnitude",default='j')
    parser.add_argument("--h",help="If loading flux, column for H magnitude",default='h')
    parser.add_argument("--k",help="If loading flux, column for K magnitude",default='k')
    parser.add_argument("--parallax",help="If loading flux, column for parallax",default='parallax')
    parser.add_argument("--eg",help="If loading flux, column for uncertainty in G magnitude",default='eg')
    parser.add_argument("--ebp",help="If loading flux, column for uncertainty in BP magnitude",default='ebp')
    parser.add_argument("--erp",help="If loading flux, column for uncertainty in RP magnitude",default='erp')
    parser.add_argument("--ej",help="If loading flux, column for uncertainty in J magnitude",default='ej')
    parser.add_argument("--eh",help="If loading flux, column for uncertainty in H magnitude",default='eh')
    parser.add_argument("--ek",help="If loading flux, column for uncertainty in K magnitude",default='ek')
    parser.add_argument("--eparallax",help="If loading flux, column for uncertainty in parallax",default='eparallax')
    parser.add_argument("--gf",help="If uncertainties have not been converted to magnitudes, column for G flux",default='phot_g_mean_flux')
    parser.add_argument("--bpf",help="If uncertainties have not been converted to magnitudes, column for BP flux",default='phot_bp_mean_flux')
    parser.add_argument("--rpf",help="If uncertainties have not been converted to magnitudes, column for RP flux",default='phot_rp_mean_flux')
    parser.add_argument("--egf",help="If uncertainties have not been converted to magnitudes, column for uncertainty in G flux",default='phot_g_mean_flux_error')
    parser.add_argument("--ebpf",help="If uncertainties have not been converted to magnitudes, column for uncertainty in BP flux",default='phot_bp_mean_flux_error')
    parser.add_argument("--erpf",help="If uncertainties have not been converted to magnitudes, column for uncertainty in RP flux",default='phot_rp_mean_flux_error')
    parser.add_argument("--memoryOnly",help="Store table only in memory without saving to disk",action='store_false')
    parser.add_argument('--ver',help="Version of Gaia data to download, default DR3",default='dr3')
    
    return parser

parser=makeparser()



#Normalizes all the variables for training, and returns back the original values afterwards
def convunit(v,b,back=False):
    a=[['g',18,0.5],['bp',21,0.5],['rp',18,0.5],['j',17.5,0.5],['h',16.5,0.5],['k',16.5,0.5],
     ['w1',16.5,0.5],['w2',16.5,0.5],['w3',14,0.5],['parallax',20,0.5],['radius',5,0.54],
     ['logl',4,0],['av',20,0.3],['age',4,2],['mass',3,0.5],['teff',0.7,3.4/0.7+0.5],
     ['logg',2,2],['feh',3,-2.5/3+0.5],['dist',3.7,0.5]]
    for i in a:
        if b==i[0]:
            if back:
                return (v+i[2])*i[1]
            else:
                return (v/i[1])-i[2]
    return bad

#https://github.com/eladhoffer/convNet.pytorch/blob/master/models/mnist.py
class Net(nn.Module):

    def __init__(self, input_shape=(1, 7, 250),drop_p=0.1):
        super(Net, self).__init__()
        self.feats = nn.Sequential(
            nn.Conv2d(1, 32, 5, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, 3,  1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, 3,  1, 1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(128)
        )

        self.classifier = nn.Conv2d(128, 10, 1)
        self.avgpool = nn.AvgPool2d(2,2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(620, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 3)

    def forward(self, inputs):
        out = self.feats(inputs)
        out = self.dropout(out)
        out = self.classifier(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)
        return out
model = Net()


def makepandasfromadql(data,args):
    from astroquery.gaia import Gaia
    if not args.silent: print('Downloading photometry from Gaia Archive')
    table=Table([data[args.cluster],data[args.source_id]],names=['cluster', 'source_id'])
    if args.ver=='dr2':
        j = Gaia.launch_job_async(query="select tc.cluster, g.source_id,g.parallax,g.parallax_error as eparallax,g.phot_g_mean_mag as g,-2.5*log10(abs(phot_g_mean_flux-phot_g_mean_flux_error))+2.5*log10(abs(phot_g_mean_flux+phot_g_mean_flux_error)) as eg, g.phot_bp_mean_mag as bp,-2.5*log10(abs(phot_bp_mean_flux-phot_bp_mean_flux_error))+2.5*log10(abs(phot_bp_mean_flux+phot_bp_mean_flux_error)) as ebp,g.phot_rp_mean_mag as rp,-2.5*log10(abs(phot_rp_mean_flux-phot_rp_mean_flux_error))+2.5*log10(abs(phot_rp_mean_flux+phot_rp_mean_flux_error)) as erp,tm.j_m as j,tm.j_msigcom as ej,tm.h_m as h,tm.h_msigcom as eh,tm.ks_m as k,tm.ks_msigcom as ek \
        FROM gaiadr2.gaia_source AS g \
        inner join TAP_UPLOAD.table_test AS tc \
        ON g.source_id = tc.source_id \
        LEFT OUTER JOIN gaiadr2.tmass_best_neighbour AS xmatch \
        ON g.source_id = xmatch.source_id \
        LEFT OUTER JOIN gaiadr1.tmass_original_valid AS tm \
        ON tm.tmass_oid = xmatch.tmass_oid", upload_resource=table, upload_table_name="table_test")
    else:
        j = Gaia.launch_job_async(query="select tc.cluster, g.source_id,g.parallax,g.parallax_error as eparallax,g.phot_g_mean_mag as g,-2.5*log10(abs(phot_g_mean_flux-phot_g_mean_flux_error))+2.5*log10(abs(phot_g_mean_flux+phot_g_mean_flux_error)) as eg, g.phot_bp_mean_mag as bp,-2.5*log10(abs(phot_bp_mean_flux-phot_bp_mean_flux_error))+2.5*log10(abs(phot_bp_mean_flux+phot_bp_mean_flux_error)) as ebp,g.phot_rp_mean_mag as rp,-2.5*log10(abs(phot_rp_mean_flux-phot_rp_mean_flux_error))+2.5*log10(abs(phot_rp_mean_flux+phot_rp_mean_flux_error)) as erp,tm.j_m as j,tm.j_msigcom as ej,tm.h_m as h,tm.h_msigcom as eh,tm.ks_m as k,tm.ks_msigcom as ek \
        FROM gaiaedr3.gaia_source AS g \
        inner join TAP_UPLOAD.table_test AS tc \
        ON g.source_id = tc.source_id \
        LEFT OUTER JOIN gaiaedr3.tmass_psc_xsc_best_neighbour AS xmatch \
        ON g.source_id = xmatch.source_id \
        LEFT OUTER JOIN gaiaedr3.tmass_psc_xsc_join AS xmatch_join \
        ON xmatch.clean_tmass_psc_xsc_oid = xmatch_join.clean_tmass_psc_xsc_oid \
        LEFT OUTER JOIN gaiadr1.tmass_original_valid AS tm \
        ON tm.designation = xmatch_join.original_psc_source_id", upload_resource=table, upload_table_name="table_test")
    d = j.get_results().to_pandas()
    #d['cluster']=table['cluster']
    if args.saveFlux !='':d.to_csv(args.saveFlux+'.csv',index=False)
    return d

def makepandasfromdata(data,args):
    if not args.silent: print('Reading in data from the table - make sure the the columns have been defined correctly')
    d=pd.DataFrame(columns=['cluster','g','bp','rp','j','h','k','age','av','dist','parallax','eparallax','eg','ebp','erp','ej','eh','ek'],index=range(len(data[args.cluster])))
    d['cluster']=np.array(data[args.cluster]).byteswap().newbyteorder()
    d['g']=np.array(data[args.g]).byteswap().newbyteorder()
    d['bp']=np.array(data[args.bp]).byteswap().newbyteorder()
    d['rp']=np.array(data[args.rp]).byteswap().newbyteorder()
    d['j']=np.array(data[args.j]).byteswap().newbyteorder()
    d['h']=np.array(data[args.h]).byteswap().newbyteorder()
    d['k']=np.array(data[args.k]).byteswap().newbyteorder()
    if args.gaiaFluxErrors:
        d['eg']=np.array((-2.5*np.log10(data[args.gf]-data[args.egf])+2.5*np.log10(data[args.gf]+data[args.egf]))/2).byteswap().newbyteorder()
        d['ebp']=np.array((-2.5*np.log10(data[args.bpf]-data[args.ebpf])+2.5*np.log10(data[args.bpf]+data[args.ebpf]))/2).byteswap().newbyteorder()
        d['erp']=np.array((-2.5*np.log10(data[args.rpf]-data[args.erpf])+2.5*np.log10(data[args.rpf]+data[args.erpf]))/2).byteswap().newbyteorder()
    else:
        d['eg']=np.array(data[args.eg]).byteswap().newbyteorder()
        d['ebp']=np.array(data[args.ebp]).byteswap().newbyteorder()
        d['erp']=np.array(data[args.erp]).byteswap().newbyteorder()
    d['ej']=np.array(data[args.ej]).byteswap().newbyteorder()
    d['eh']=np.array(data[args.eh]).byteswap().newbyteorder()
    d['ek']=np.array(data[args.ek]).byteswap().newbyteorder()
    d['parallax']=np.array(data[args.parallax]).byteswap().newbyteorder()
    d['eparallax']=np.array(data[args.eparallax]).byteswap().newbyteorder()
    return d

def fillmissing(d):
    d['bp']=d['bp'].fillna(21)
    d['ebp']=d['ebp'].fillna(0)
    d['rp']=d['rp'].fillna(18)
    d['erp']=d['erp'].fillna(0)
    d['j']=d['j'].fillna(17.5)
    d['ej']=d['ej'].fillna(0)
    d['h']=d['h'].fillna(16.5)
    d['eh']=d['eh'].fillna(0)
    d['k']=d['k'].fillna(16.5)
    d['ek']=d['ek'].fillna(0)
    d['eparallax']=d['eparallax'].fillna(0.2)
    
    a=np.argsort(d['cluster'])
    dd=d.iloc[a]
    return dd

def maketensor(d,args):
    if not args.silent: print('Putting together a tensor file')
    un=np.unique(d['cluster'])
    clusterx=torch.zeros(len(un)*args.iters, 1,7, 250)*0+0.5
    clusterx[:,:,6]=clusterx[:,:,6]-1
    clusterref=pd.DataFrame(columns=['cluster'],index=range(len(un)*args.iters))
    n=0
    for i in un:
        temp=d[(d['cluster']==i)]
        try:
            i=i.decode()
        except:
            pass
        if not args.silent: print(i)
        
        x=len(temp)
        for todo in range(args.iters):
            rand=np.random.normal(0,1,x*7)
            sel=temp.copy()
            
            sel['g']=sel['g']+rand[0:x]*sel['eg']
            sel['bp']=sel['bp']+rand[x:2*x]*sel['ebp']
            sel['rp']=sel['rp']+rand[2*x:3*x]*sel['erp']
            sel['j']=sel['j']+rand[3*x:4*x]*sel['ej']
            sel['h']=sel['h']+rand[4*x:5*x]*sel['eh']
            sel['k']=sel['k']+rand[5*x:6*x]*sel['ek']
            sel['parallax']=sel['parallax'].to_numpy()+rand[6*x:7*x]*sel['eparallax']
            
            a=np.random.choice(np.arange((len(sel))),250)
            sel=sel.iloc[a]
            sel=sel.sort_values(by=['g'])
            clusterx[n][0][0][0:len(a)]=torch.Tensor(convunit(sel['g'],'g').to_numpy())
            clusterx[n][0][1][0:len(a)]=torch.Tensor(convunit(sel['bp'],'bp').to_numpy())
            clusterx[n][0][2][0:len(a)]=torch.Tensor(convunit(sel['rp'],'rp').to_numpy())
            clusterx[n][0][3][0:len(a)]=torch.Tensor(convunit(sel['j'],'j').to_numpy())
            clusterx[n][0][4][0:len(a)]=torch.Tensor(convunit(sel['h'],'h').to_numpy())
            clusterx[n][0][5][0:len(a)]=torch.Tensor(convunit(sel['k'],'k').to_numpy())
            clusterx[n][0][6][0:len(a)]=torch.Tensor(convunit(sel['parallax'],'parallax').to_numpy())   
            clusterref['cluster'][n]=i
            n=n+1
    #if save!='': pickle.save([clusterx,clusterref],open('final31.pickle','wb'))
    return clusterx,clusterref

def predict(clusterx,clusterref,name,args,iters=10):
    if not args.silent: print('Predicting population properties')
    model.load_state_dict(torch.load(pkg_resources.resource_filename('auriga', 'auriga.pt'), map_location='cpu'))
    model.eval()

    result=pd.DataFrame(index=clusterref.index,columns=['age','av','dist'])
    k=np.array(range(len(clusterx)))
    batch_size=5000
    for i in range(0, len(clusterx), batch_size):
        inputs = clusterx[k[i:i + batch_size]]
        with torch.no_grad():
            a=model(inputs)
        r=a.cpu().detach().numpy()
        result['age'].iloc[k[i:i + batch_size]]=convunit(r[:,0],'age',back=True)
        result['av'].iloc[k[i:i + batch_size]]=convunit(r[:,1],'av',back=True)
        result['dist'].iloc[k[i:i + batch_size]]=convunit(r[:,2],'dist',back=True)
    
        
    r=pd.DataFrame(columns=['cluster','age','eage','av','eav','dist','edist'],index=np.arange(len(result)/iters))
    for i in np.arange(len(result)/iters).astype(int):
        r['age'].iloc[i]=np.mean(result['age'].iloc[i*iters:i*iters+iters])
        r['eage'].iloc[i]=np.std(result['age'].iloc[i*iters:i*iters+iters])
        r['av'].iloc[i]=np.mean(result['av'].iloc[i*iters:i*iters+iters])
        r['eav'].iloc[i]=np.std(result['av'].iloc[i*iters:i*iters+iters])
        r['dist'].iloc[i]=np.mean(result['dist'].iloc[i*iters:i*iters+iters])
        r['edist'].iloc[i]=np.std(result['dist'].iloc[i*iters:i*iters+iters])
        r['cluster'].iloc[i]=clusterref['cluster'].iloc[i*iters]
    
    r['dist']=10**r['dist']
    r['edist']=r['edist']*r['dist']*np.log(10)
    r=Table.from_pandas(r)
    k=r.keys()
    for i in range(1,len(k)):
        r[k[i]]=r[k[i]].astype(float)
        
    r['age'],r['eage']=roundtable(r['age'],r['eage'])
    r['dist'],r['edist']=roundtable(r['dist'],r['edist'])
    r['av'],r['eav']=roundtable(r['av'],r['eav'])

    return r

def roundtable(value,error):
    for i in range(len(value)):
        try:
            try:
                value[i]=round(value[i],round(1-np.floor(np.log10(error[i]))))
            except:
                value[i]=round(value[i],round(1-np.floor(np.log10(value[i]*0.05))))
            error[i]=round(error[i],round(1-np.floor(np.log10(error[i]))))
        except:
            pass
    return value,error

def tolowercase(data):
    keys=data.keys()
    for key in keys: data[key].name=key.lower()
    return data
    
def getClusterAge(into=None,args=None,data=None,tableIn=None,tutorial=None,tableOut=None,iters=None,localFlux=None,saveFlux=None,silent=None,cluster=None,source_id=None,
         gaiaFluxErrors=None,g=None,bp=None,rp=None,j=None,h=None,k=None,parallax=None,eg=None,ebp=None,erp=None,ej=None,eh=None,ek=None,eparallax=None,
         gf=None,bpf=None,rpf=None,egf=None,ebpf=None,erpf=None,memoryOnly=None,ver=None):
    if type(into)==str: tableIn=into
    if type(into)==Table: data=into
    if args is None:
        args=parser.parse_args([])
        if tableIn is not None: 
            args.tableIn=tableIn
        else: 
            args.tableOut='test'
        if tutorial is not None: args.tutorial=tutorial
        if tableOut is not None: args.tableOut=tableOut
        if iters is not None: args.iters=iters
        if localFlux is not None: args.localFlux=localFlux
        if saveFlux is not None: args.saveFlux=saveFlux
        if silent is not None: args.silent=silent
        if cluster is not None: args.cluster=cluster
        if source_id is not None: args.source_id=source_id
        if gaiaFluxErrors is not None: args.gaiaFluxErrors=gaiaFluxErrors
        if g is not None: args.g=g
        if bp is not None: args.bp=bp
        if rp is not None: args.rp=rp
        if j is not None: args.j=j
        if h is not None: args.h=h
        if k is not None: args.k=k
        if parallax is not None: args.parallax=parallax
        if eg is not None: args.eg=eg
        if ebp is not None: args.ebp=ebp
        if erp is not None: args.erp=erp
        if ej is not None: args.ej=ej
        if eh is not None: args.eh=eh
        if ek is not None: args.ek=ek
        if eparallax is not None: args.eparallax=eparallax
        if gf is not None: args.gf=gf
        if bpf is not None: args.bpf=bpf
        if rpf is not None: args.rpf=rpf
        if egf is not None: args.egf=egf
        if ebpf is not None: args.ebpf=ebpf
        if erpf is not None: args.erpf=erpf
        if memoryOnly is not None: args.memoryOnly=memoryOnly
        if ver is not None: args.ver=ver

    if data is None:
     if args.tableIn=='':
         parser.print_help(sys.stderr)
         sys.exit(1)
     
     if args.tutorial: 
         dr=pkg_resources.resource_filename('auriga', 'test/'+args.tableIn)
         if args.silent:print('Reading test file, '+args.tableIn)
     else: 
         dr=args.tableIn
     if args.tableOut=='': args.tableOut=args.tableIn.split('.')[0]+'-out'
     if(os.path.exists(dr)):
         data = tolowercase(Table.read(dr))
     else:
         raise ValueError("Can't find the input table")
         
    if args.localFlux:
        d=fillmissing(makepandasfromdata(data,args))
    else:
        d=fillmissing(makepandasfromadql(data,args))
        
    clusterx,clusterref=maketensor(d,args)
    r=predict(clusterx,clusterref,args.tableOut,args,iters=args.iters)
    

    if args.memoryOnly:
        if not args.silent:
            print(r)
            print('Written out to '+args.tableOut+'.fits')
        r.write(args.tableOut+'.fits',overwrite=True)
    return r
    
def main():
    args=parser.parse_args()
    getClusterAge(args=args)
    
if __name__ == '__main__':
    main(args)