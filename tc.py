'''
This module contains several classes for managing the conversion between different time representations in the MGCM:

    - tc: time conversion object with several useful functions/methods for time conversion between 5 different time formats    
    - time: time object that allows for (operator-overloaded) arithmetic time manipulation in each of the 5 formats

There is also a command line facility at the end of the script, that allows time conversions to be performed on the command line.        
'''

import numpy as np
from scipy.optimize import root
import os
import warnings

class tc:
    '''
    A class for managing time in the OU Mars Global Climate Model (MGCM). Supports conversion between five different types of time format.
    Supported time formats:
            'soll':  'Extended sol' number: the (real) number of sols that have elapsed since midnight on sol=0 of MY24. 
            'sol':   Number of sols since the start of a given martian year. 
            'ls':    The solar longitude in a given martian year.
            'index': The folder name and array index of a time point from the MGCM output.
            'utc'  : The time on Earth in Universal Coordinated Time.
    
    Methods:     
        tc.MY_blocks: returns the list of MGCM run blocks and start/end indices that include all the time points for a given Mars Year.
        tc.convert: converts between different time formats 'soll', 'sol', 'ls', 'index' and 'utc'.
        tc.soll_blocks: returns the list of MGCM run blocks and start/end indices that include all the time points between two given soll points (inclusive)
        tc.soll_info: reports on some of the details about a particular (possibly fractional) soll number.
        tc.dayshift: converts between local times at different longitudes.
        
    For more information about each method, look at the docstrings (e.g. `tcx.convert?` in ipython console).

    Initialisation:
        tcx = tc(timespersol,blocklength)
        
        Inputs:
            timespersol: The number of time points that are output over each sol in the MGCM run (usually=12 or 24, but depends on the run).
            blocklength:The number of sols that make up a MGCM run block (usually=30, but depends on the run).
    

    contact: kylash.rajendran@open.ac.uk
    '''
    def __init__(self,timespersol,blocklength):
        self.timespersol=timespersol
        self.blocklength=blocklength
        self._year_day=668.6 # 668.5921
        self._day_s=88775.245
        self._peri_day=485.0
        self._timeperi=1.905637
        self._e_elips=0.093358
        self._degrad=57.295779
        self._offset_year=24
        self._jdate_ref = 2.451009295243e8 # reference julian date (19:05:09 on 14/07/1998), which is Ls=0 for MY24.
        self._soll_offset=0.2485534 # 05:57:55, corresponding local time at Airy-0 when Ls=0, MY24.        
        self._earthday=86400.0  
        self.filestartday=1 # the output naming convention for sols in the model blocks (i.e is time=0.0 represented as midnight of sol 0 or sol 1 in the model naming convention)
        self._indexoffset=-1 # this is the indexing offset of midnight, relative to the first recorded time entry of soll 0. E.g. if first recorded entry is 0.0833333, for timespersol=12, then set _indexoffset=-1, since midnight is one time unit behind first entry)
        self._num_dp=10
        
        # table of leap seconds, last updated on 27 June 2021    
        self._leap_list = [
            [1972,6,30,23,59,59],
            [1972,12,31,23,59,59],
            [1973,12,31,23,59,59],
            [1974,12,31,23,59,59],
            [1975,12,31,23,59,59],    
            [1976,12,31,23,59,59],
            [1977,12,31,23,59,59],
            [1978,12,31,23,59,59],
            [1979,12,31,23,59,59],
            [1981,6,30,23,59,59],
            [1982,6,30,23,59,59],
            [1983,6,30,23,59,59],
            [1985,6,30,23,59,59],
            [1987,12,31,23,59,59],
            [1989,12,31,23,59,59],
            [1990,12,31,23,59,59],
            [1992,6,30,23,59,59],    
            [1993,6,30,23,59,59],
            [1994,6,30,23,59,59],
            [1995,12,31,23,59,59],
            [1997,6,30,23,59,59],
            [1998,12,31,23,59,59],
            [2005,12,31,23,59,59],
            [2008,12,31,23,59,59],
            [2012,6,30,23,59,59],
            [2015,6,31,23,59,59],
            [2016,12,31,23,59,59]
            ]

    ##########################################################
    # convert: A generalized time conversion function        
    def convert(self,type1,type2,val1,val2=None,approx='closest'):
        '''
        convert(type1,type2,val1,val2=None): A function to convert between different time formats.
        
        Format types:
            'soll':  
                'Extended sol' number: the (real) number of sols that have elapsed since midnight on sol=0 of MY24. 
                Defined by a single float (val1=soll) where soll range is [0,\infty).
            'sol':   
                Number of sols since the start of a given martian year. 
                Defined by a float and int combination (val1=sol,val2=MY) where sol range is [0,669) and MY range is [24,\infty).                
                The year length is either 668 or 669 sols, depending on the year (see note below for details).
            'ls':    
                The solar longitude based on a martian year length of exactly 668.6 sols.
                Defined by a float and int combination (val1=ls,val2=MY) where ls range is [0-,360+] and MY range is [24,\infty).
                The range of ls is allowed to expand slightly beyond [0,360] on either side, in order to maintain consistency of MY definitions (see note below for details).
    
            'index': 
                The folder name and array index of a time point from the MGCM output.
                Defined by a string and float combination (val1=folder,val2=index) where folder is the MGCM folder output name and index is a particular time index in the output array (fractional index values are permitted).
                
            'utc':
                Earth time in UTC.
                Defined by an array of 6 integers: (val1=[year,month,day,hour,minute,second])
        
        Inputs:
            type1: String giving the time format to convert from. Any of the above format types ('soll','sol','ls','index','utc') is allowed.
            type2: String giving the time format to convert to. Any of the above format types ('soll','sol','ls','index','utc') is allowed.
            val1,val2: The time details of the format to convert from, which varies depending on the format:
                - type1='soll':  val1 is the soll number (e.g. 7021.2), val2 is unnecessary.
                - type1='sol':   val1 is the sol number (e.g. 660.3), val2 is the MY of interest (e.g. 34).
                - type1='ls':    val1 if the ls degree (e.g. 180), val2 is the MY of interest (e.g. 34).
                - type1='index': val1 is the folder name as a string (e.g. 'd7021-7051'), val2 is a particular array index (e.g. 0 for the first time point in the model block).
                - type1='utc':   val1 is a 6-element list [year,month,day,hour,minute,second], val2 is unnecessary.
            approx: Indicates how to deal with conversions to model indices:
                'closest': returns closest index
                'before' : returns closest index before requested point
                'after' : returns closest index after requested point
        Outputs:
            If type2='sol','ls','index': output is a tuple (val1,val2), with time formatting described as above. 
            If type2='soll', output is a float.
            If type2='utc', output is a 6-element array.
            
        Notes:
            
            - Conversions between utc <-> Ls <-> soll are based on the calculations of Allison & McEwen 2000, A post-Pathfinder evaluation of aerocentric solar coordinates with improved timing recipes for Mars seasonal/diurnal climate studies. Planet. Space Sci., 48, p215-235; as well as the javascript implementation of the above by James Tauber (www.marsclock.com). These calculations give accuracy of Ls to 0.01 degrees.

            - The conversion from Ls -> utc is based on a numerical inversion. The initial guess is provided by the original IDL ls -> sol implementation.

            - In general, conversions will be accurate to within half a martian hour. For detailed work (e.g. data assimilation), it is still best to get the local time from the observation data.
            
            - Conversion to index format results in loss of information, since the (integer) model data indices have limited time resolution.
    
            - 'solls' is the count of days starting from soll=0.0 which is midnight of the sol on which MY24 begins (this is not the same time as Ls=0 of MY24, which occurs a little later at 05:57:55, or soll=0.25)

            - folder, index is specific to the MGCM, and refers to a (0-based) index in a particular model output folder;
            e.g: folder = 'd1-31', index=0 refers to midnight on sol 1 (ie time 0.0) of MY24
            Note that the numbers in the folder string are days based on 'January 1st'-type counting. For example, in d1-31 the '1' refers to the sol covering the period soll=0.083-1.0, and the '31' covers the period soll=30.083-31.0.

        Notes on defining Mars year lengths: 

            - Mars years are assigned different (integer) lengths by rounding, to account for the fractional 668.6-sol length of an orbital martian year. 
            - The pattern of year lengths repeats every five years (since 668.6*5=3343 is an integer). 
            - For MYs 24,25,26,27,28 the lengths are 669,668,669,668,669 sols respectively. The same pattern of year lengths then repeats for MYs 29,30,31,32,33, and so on.

            - The convert function will always ensure that the correct MY is assigned for a given time point. 
                e.g. if user gives sol=668.7,MY=25, convert will recognise this time point as belonging to MY26 and will return values accordingly:
                    `convert('sol','soll',668.5,25)` returns `1337.5`
                    `convert('soll','sol',1337.5)` returns `(0.5, 26)`                        

            - In order to make this naming scheme consistent, ls is not required to strictly lie between [0,360]; on the edge sols between years, ls is allowed to go negative or past 360 in order to maintain the year naming convention
                e.g. `convert('sol','ls',668,25)` returns `(-0.21998, 26)`

        '''        
        # Error checking
        if type1 == type2:
            raise Exception('type1 and type2 cannot be equal.')        
        if (type1=='sol' or type1=='ls') and val2 == None:
            raise Exception('Conversions from '+type1+' requires a MY to be specified as an argument.')    
        if type1 == 'utc':
            self._utc_check(val1)
        if type1 == 'sol' and type2 == 'ls':
            ls,MY = self._sol_lsMY(val1,val2)
            return ls, MY
        if type1 == 'sol' and type2 == 'soll':
            soll = self._sol_soll(val1,val2)
            return soll   
        if type1 == 'sol' and type2 == 'index':
            folder, index = self._sol_index(val1,val2,approx=approx)
            return folder, index    
        if type1 == 'sol' and type2 == 'utc':
            utc = self._sol_utc(val1,val2)
            return utc
        if type1 == 'ls' and type2 == 'sol':
            sol, MY = self._ls_solMY(val1,val2)
            return sol, MY    
        if type1 == 'ls' and type2 == 'soll':
            soll = self._ls_soll(val1,val2)
            return soll  
        if type1 == 'ls' and type2 == 'index':
            folder, index = self._ls_index(val1,val2,approx=approx)
            return folder, index    
        if type1 == 'ls' and type2 == 'utc':
            utc = self._ls_utc(val1,val2)
            return utc
        if type1 == 'soll' and type2 == 'sol':
            sol, MY = self._soll_sol(val1)
            return sol, MY    
        if type1 == 'soll' and type2 == 'ls':
            ls, MY = self._soll_ls(val1)
            return ls, MY    
        if type1 == 'soll' and type2 == 'index':
            folder, index = self._soll_index(val1,approx=approx)
            return folder, index    
        if type1 == 'soll' and type2 == 'utc':
            utc = self._soll_utc(val1)
            return utc
        if type1 == 'index' and type2 == 'sol':
            sol, MY = self._index_sol(val1,val2)
            return sol, MY    
        if type1 == 'index' and type2 == 'ls':
            ls, MY = self._index_ls(val1,val2)
            return ls, MY    
        if type1 == 'index' and type2 == 'soll':
            soll = self._index_soll(val1,val2)
            return soll
        if type1 == 'index' and type2 == 'utc':
            utc = self._index_utc(val1,val2)
            return utc
        if type1 == 'utc' and type2 == 'sol':
            sol, MY = self._utc_sol(val1)
            return sol, MY    
        if type1 == 'utc' and type2 == 'ls':
            ls, MY = self._utc_ls(val1)
            return ls, MY    
        if type1 == 'utc' and type2 == 'soll':
            soll = self._utc_soll(val1)
            return soll
        if type1 == 'utc' and type2 == 'index':
            folder, index = self._utc_index(val1,approx=approx)
            return folder, index    
        # If we reach this point, something has gone wrong
        raise Exception('At least one of type1='+type1+', type2='+type2+' is not recognised. Accepted values are `sol`, `ls`, `soll`, `index`.')
        
    ##########################################################
    
    def soll_info(self,soll):
        '''
         soll_info(soll): A function that reports on some of the details about a particular (possibly fractional) soll number
         
         Inputs: 
             - soll: the extended sol number, as used by the model. This counts the number of sols since the start of MY24
           
        Outputs: None
        
       '''
#        soll = np.round(soll,decimals=self._num_dp)
        if soll < 0:
            raise Exception('soll must be >=0')
        
        # get the Mars year and sol number
        sol, MY = self._soll_sol(soll)
        # get the Ls number
        ls = self._soll_ls(soll)[0]
        # get the folder and index information
        folder, index = self._soll_index(soll)
        utc = self._soll_utc(soll)
        
        print('******************************')        
        print('Model soll number: '+str(soll))
        print('Mars Year: '+str(MY))
        print('sol number: '+str(sol))
        print('Ls: '+str(ls))
        print('sol block: '+folder)
        print('Time index (Python 0-based): '+str(index))        
        print('UTC time: '+str(utc))
        print('******************************')
        
    ##########################################################
   
    def soll_blocks(self,startsoll,endsoll,folderStr='',ncStr='',approx='closest'):
        '''
        soll_blocks(startsoll,endsoll,folderStr='',ncStr=''): A function that returns a list of the set of soll blocks that would include
        all the time points between two given soll points (inclusive).
        
        Inputs: 
            - startsoll: the starting soll point of interest (float)
            - endsoll: the ending soll point of interest (float)
            - folderStr='': parent folder of the model outputs (string)
            - ncStr='': name of the NetCDF file containing the model outputs (string)
          
        Outputs:
            - blocklist: array. Each entry is an array with three elements: [blockStr, startInd, endInd], where:
                blockStr: the name of the folder block (string)
                startInd: the index of the first relevant time index in that block
                endInd: the index of the last relevant time index in that block
       '''
        # get the start block and start index for the year
        sblockStr, sInd = self.convert('soll','index',startsoll,approx=approx); sInd = int(sInd)
        # get the end block and end index for the year
        eblockStr, eInd = self.convert('soll','index',endsoll,approx=approx); eInd = int(eInd)
        # get the start and end block names as integers
        sblock = int(sblockStr[1:sblockStr.index('-')])
        eblock = int(eblockStr[1:eblockStr.index('-')])
        # Calculate the number of blocks
        nblocks = int((eblock-sblock)/self.blocklength+1)
        firstInd = 0; lastInd = int(self.timespersol*self.blocklength-1)
        # Now construct the list of soll blocks
        blocklist = [[0 for j in range(3)] for i in range(nblocks)]
        if nblocks == 1:
            blocklist[0] = [sblockStr, sInd, eInd]
        else:
            blocklist[0] = [sblockStr, sInd, lastInd]
            for i in np.arange(1,nblocks):
                strStart = str(int(sblock+i*self.blocklength)); strEnd = str(int(sblock+(i+1)*self.blocklength))
                blocklist[i] = ['d'+strStart+'-'+strEnd, firstInd, lastInd]
            blocklist[-1] = [eblockStr, firstInd, eInd]
        if folderStr != '':
            for row in blocklist:
                row[0] = os.path.join(folderStr,row[0])
        if ncStr != '':
            for row in blocklist:
                row[0] = os.path.join(row[0],ncStr)
        return blocklist
    
    ##########################################################
    
    def MY_blocks(self,MY,folderStr='',ncStr=''): 
        '''
        MY_blocks(MY,folderStr='',ncStr=''): A function that returns a list of the set of soll blocks that would include all the time points for a given Mars Year.
        Inputs:
            - MY: Mars year of interest
            - folderStr='': string giving the parent folder of the model outputs
            - ncStr='': string giving the name of the NetCDF file containing the model outputs

        Outputs:
            - blocklist: array. Each entry is an array with three elements: [blockStr, startInd, endInd], where:
                blockStr: the name of the folder block (string)
                startInd: the index of the first relevant time index in that block
                endInd: the index of the last relevant time index in that block
      '''
        startsoll,endsoll = self._MY_bds(MY)
        blocklist = self.soll_blocks(startsoll,endsoll)#-1./self.timespersol)
        return blocklist

    ##########################################################
    # function to convert time differences between different longitudes
    def dayshift(self,sol1,t1,lon1,lon2,infrac=True,outfrac=True):
        '''
        Function to convert between local times at different longitudes.
        Inputs:
          sol1:  integer: sol/soll number of source location
          t1:    float: hour / dayfraction at source longitude (hour if frac=False, dayfraction if frac=True)
          lon1:  float: source longitude [0,360) or [-180,180)
          lon2:  float: destination longitude [0,360) or [-180,180)
          infrac:  logical determining whether t1 is interpreted as an hour or a dayfraction (hour if infrac=False, dayfraction if infrac=True)
          outfrac: logical determining whether t2 is interpreted as an hour or a dayfraction (hour if outfrac=False, dayfraction if outfrac=True)
        Outputs:
          sol2 - integer: sol/soll number of source longitude
          t2 - hour / dayfraction at destination longitude (hour if outfrac=False, dayfraction if outfrac=True)
        
        '''
        
        # shift both lons to be in range [0,360)
        lon1=lon1%360
        lon2=lon2%360     
        
        if t1<0:
            raise ValueError('day fraction must be positive.')
        
        # convert time to hour format
        if infrac == True: # day fraction format
            t1 = np.round(24*t1,decimals=self._num_dp) # local time in hours at source
        else: # already in hour format
            pass

        # get midnight longitude
        if t1 <=12:
            mlon = (lon1-(t1*15))%360
        else:
            mlon = (lon1+15*(24-t1))%360
        
        if mlon <= 180: # new day is for longitudes between mlon and 180
            if lon1>=mlon and lon1<=180: # lon1 is in new day
                if lon2>=mlon and lon2<=180: # lon2 is also in new day
                    sol2=sol1                    
                else: # lon2 is in old day
                    sol2=sol1-1
            else: # lon1 is in old day
                if lon2>=mlon and lon2<=180: # lon2 is in new day
                    sol2=sol1+1
                else: # lon2 is also in old day
                    sol2=sol1
        else: # mlon > 180; old day is for longitudes between 180 and mlon
            if lon1>180 and lon1<mlon: #lon1 is in old day
                if lon2>180 and lon2<mlon: #lon2 is also in old day
                    sol2=sol1
                else: # lon2 is in new day
                    sol2=sol1+1
            else: # lon1 is in new day
                if lon2>180 and lon2<mlon: #lon2 is in old day
                    sol2=sol1-1
                else: # lon2 is also in new day
                    sol2=sol1
        if sol2<0:
            Warning('Requested local time is before first model time point. Using first day instead.')
            sol2 = 0
                    
        # get time at lon2
        t2 = (t1+(lon2-lon1)/15)%24
        if outfrac is True:
            t2/=24
        
        return sol2,t2
    
    ##########################################################
    # Utility function to return the day boundaries [a,b) for a given MY. 
    def _MY_bds(self,MY):        
        a = float(MY-self._offset_year+1)
        startday = int(np.round((a-1)*self._year_day))#+1)
        endday = int(np.round(a*self._year_day))    
        return startday, endday
               
    ##########################################################
    # function to convert from sol to soll;
    # here soll is the sol number count starting from MY24
    
    def _sol_soll(self,sol,MY,warn=False):
        if sol < 0:
            raise Exception('sol must not be negative.')
        startsoll, endsoll = self._MY_bds(MY)
        soll = np.round(startsoll+sol,decimals=self._num_dp)
        #soll = np.round(sol+self._year_day*(MY-self._offset_year),decimals=self._num_dp) # should we be using this version instead? KR, 09 2021
        # check that soll is within MY bds        
        if soll >= endsoll and warn==True:
            print('Warning: Input sol '+str(soll)+' is outside of the range of MY'+str(MY)+': ['+str(startsoll)+', '+str(endsoll)+')')
        return soll
        
    ##########################################################
    # function to convert from soll to sol
    # here soll is the sol number count starting from MY24
    # Outputs:
    #   - sol: sol in the appropriate range for the Mars Year it falls within
    #   - MY: the corresponding Mars Year
        
    def _soll_sol(self,soll):      
        # Get the Ls cycle of this soll
        #k = int(soll/self._year_day)
        k = np.floor(soll/self._year_day)
        # get start and end solls for this Ls cycle
        Ls0_firstsoll = np.round(self._year_day*(k),decimals=self._num_dp)
        Ls0_lastsoll = np.round(self._year_day*(k+1),decimals=self._num_dp)
        # calculate sol fraction of the boundary solls at point when Ls=0
        firstrem = np.round(np.modf(Ls0_firstsoll)[0],decimals=self._num_dp)
        lastrem = np.round(np.modf(Ls0_lastsoll)[0],decimals=self._num_dp)
        # Assign MY accordingly
        if soll < np.ceil(Ls0_firstsoll) and firstrem >= 0.5: # soll belongs to the previous year's Ls cycle
            MY = self._offset_year+k-1
        elif soll >= np.floor(Ls0_lastsoll) and lastrem < 0.5: # soll belongs to the next year's Ls cycle
            MY = self._offset_year+k+1
        else: # soll belongs to the current year's Ls cycle
            MY = self._offset_year+k
        sol = soll - np.round(self._year_day*(MY-self._offset_year))
        sol = np.round(sol,decimals=self._num_dp) # round to specified accuracy
        #print('_soll_sol: '+str(sol))
        return sol,MY
    
    ##########################################################
    # function to convert from soll to ls
    # here soll is the sol number count starting from MY24

    def _soll_ls(self,soll):
        utc = self._soll_utc(soll)
        julian = self._utc_julian(utc)
        ls,MYtemp = self._julian_ls(julian)
        
        # check that the year and ls defintions are consistent with the year boundaries
        sol,MY = self._soll_sol(soll)
        
        if MYtemp == MY: # the year defintions agree
            return ls, MY
        else: # the year defintions do not agree
            if MYtemp < MY: 
                ls -= 360
                return ls,MY
            else : #MYtemp > MY
                ls+=360
                return ls,MY
            
    ##########################################################
    # function to convert from ls to soll;
    # here soll is the sol number count starting from MY24
     
    def _ls_soll(self,ls,MY):
        julian = self._ls_julian(ls,MY)
        utc = self._julian_utc(julian)
        soll = self._utc_soll(utc)
        return soll

    ##########################################################
    # function to convert from sol to ls, centered around a particular MY
    def _sol_lsMY(self,sol,MY):
#        sol = np.round(sol,decimals=self._num_dp)
        soll = self._sol_soll(sol,MY)
        ls,MY = self._soll_ls(soll)
        return ls, MY
    
    ##########################################################
    # function to convert from ls to sol, centered around a particular MY
    def _ls_solMY(self,ls,MY):
        soll = self._ls_soll(ls,MY)
        sol, MY = self._soll_sol(soll)
        sol = np.round(sol,decimals=self._num_dp) # round to specified accuracy
        #print('_ls_sollMY: '+str(sol))
        return sol, MY
   
    ##########################################################
    # function to convert from soll to index
    # here soll is the sol number count starting from MY24
    # Inputs:
    #   - soll: the sol number count of the time point, starting from MY24
    # Outputs:
    #   - folder: string of the model output folder containing the soll
    #   - index: float of the index of the desired soll
    
    def _soll_index(self,soll,approx='closest'):                
        if np.round(soll,self._num_dp) < np.round(-self._indexoffset/self.timespersol,self._num_dp): 
            print('Warning: tc._soll_index: Requested soll is before first model time point. Lowest allowed value is soll='+str(1./self.timespersol)+'. I am returning first model point.')                            
            block1 = self.filestartday; index = 0
        else:
            tstepsfrom0 = self.timespersol*soll
            if approx == 'closest' or approx == '<>' or approx == '><':
                totalind = int(np.round(tstepsfrom0))+self._indexoffset
            elif approx == 'before' or approx == '<=':
                totalind = int(np.floor(tstepsfrom0))+self._indexoffset
            elif approx == 'after' or approx == '>=':
                totalind = int(np.ceil(tstepsfrom0))+self._indexoffset            
            else:
                print('Warning: approx=\''+str(approx)+'\' is not recognised. Only {\'closest\', \'<>\', \'><\', \'before\', \'<=\', \'after\', \'>=\'} are allowed values. Switching to approx=\'closest\'.')
                totalind = int(np.round(tstepsfrom0))+self._indexoffset
            # get folder number
            numfol = int(totalind/(self.timespersol*self.blocklength))+1
            block1 = self.blocklength*(numfol-1)+self.filestartday
            index = totalind-(numfol-1)*self.timespersol*self.blocklength
                
        folder = 'd'+str(block1)+'-'+str(block1+self.blocklength)
        return folder, index
    
    ##########################################################
    # function to convert from index to sol
    # Inputs:
    #   - folder: string of the model output block containing the time point of interest
    #   - index: index of the time point in question within a model block (can be an int or a float)
    # Outputs:
    #   - soll: the sol number count of the time point, starting from MY24   
    
    def _index_soll(self,folder,index):    
        if index <0 or index >= self.timespersol*self.blocklength:
            raise Exception('index is out of the range [0, '+str(self.timespersol*self.blocklength)+').')    
        # get the start time of this block
        block1 = int(folder[folder.index('d')+1:folder.index('-')])-1
        # Now calculate the soll value
        soll = np.round(block1+(index-self._indexoffset)/self.timespersol,decimals=self._num_dp)
        #print('_index_soll: '+str(soll))
        return soll
    
    ##########################################################
    # function to convert from ls to index
    
    def _ls_index(self,ls,MY,**kwargs):
        soll = self._ls_soll(ls,MY)
        folder, index = self._soll_index(soll,**kwargs)    
        return folder, index
    
    ##########################################################
    # function to convert from index to ls
        
    def _index_ls(self,folder,index):
        soll = self._index_soll(folder,index)
        ls, MY = self._soll_ls(soll)
        return ls, MY
    
    ##########################################################
    # function to convert from sol to index
        
    def _sol_index(self,sol,MY,**kwargs):
        soll = self._sol_soll(sol,MY)
        folder, index = self._soll_index(soll,**kwargs)    
        return folder, index
    
    ##########################################################
    # function to convert from index to sol
        
    def _index_sol(self,folder,index):
        soll = self._index_soll(folder,index)
        sol, MY = self._soll_sol(soll)
        return sol, MY

    ##########################################################
    # function to convert from utc to julian time
    def _utc_julian(self,utc):
    
        #ref_year=1968
        #ref_jdate=2.4398565e6 # Julian date for 01/01/1968 00:00:00      
        
        ref_year = 1970
        ref_jdate=2.4405875e6 # Julian date for 01/01/1970 00:00:00 (the Unix epoch)     
        
        # create (12-element) array of cumulative days elapsed days prior to the current month (non-leap years)
        edays = np.array([0,31,59,90,120,151,181,212,243,273,304,334])
        
        year=utc[0]
        month=utc[1]
        day=utc[2]
            
        # compute number of days from ref_year to the year of interest
        nday=0.0 # number of days
        if year>ref_year: # after reference year
            for i in np.arange(ref_year,year):
                nday=nday+365.0
                # account for leap years
                if self._checkleap(i):
                    nday = nday + 1
        else: # before reference year
            for i in np.arange(year,ref_year):
                nday=nday-365.0            
                # account for leap years
                if self._checkleap(i):
                    nday = nday - 1
        
        # add number of days due to elapsed months
        nday=nday+edays[month-1]
        # add 1 extra day if year of interest is a leap year, and date is past February
        if self._checkleap(year) and month>=3:
            nday=nday+1
        
        # add reference year offset and day    
        #jdate=ref_jdate+nday+day-1.0
        jdate=nday+day+ref_jdate-1.0
    
        # add time (hours+minutes+seconds)
        hour=utc[3]
        minute=utc[4]
        second=utc[5]    
        jdate=jdate+hour/24.0+minute/1440.0+second/86400.0
    
        return jdate

    ##########################################################
    # function to convert from julian time to utc
    def _julian_utc(self,jdate):
        
        #ref_year=1968
        #ref_jdate=2.4398565e6 # Julian date for 01/01/1968 00:00:00      
        
        ref_year = 1970
        ref_jdate=2.4405875e6 # Julian date for 01/01/1970 00:00:00 (the Unix epoch)        
        
        # create (12-element) array of cumulative days elapsed days prior to the current month (non-leap years)
        edays = np.array([0,31,59,90,120,151,181,212,243,273,304,334])        
        edays_leap = np.array([0,31,60,91,121,152,182,213,244,274,305,335]) 

        jdate = jdate - ref_jdate#+1 # this is number of days from start of ref_year, including day fraction
        #print(jdate)
        
        # split jdate into 'day' and 'time' components
        jdays = int(np.floor(jdate)) # number of days since the start of the day of interest and the reference day
        jtime = int(np.round(86400*(jdate-jdays))) # total number of seconds elapsed in the day
        #print(jdays)
        
        # split jtime into seconds, minutes, hours
        second = jtime-60*int(jtime/60)
        jtime = int(jtime/60) # total number of minutes elapsed in the day
        minute = jtime-60*int(jtime/60)
        hour = int(jtime/60)
        
        # count number of years to reference year
        year=ref_year
        if jdays < 0: # before reference year
            count = 0
            while count > jdays:
                year = year - 1
                if self._checkleap(year):
                    count = count - 366.0
                else:
                    count = count - 365.0
            jdays = jdays - count # number of days from start of year of interest
        else: # at reference year, or after
            count = 0
            thisyear = 365.0+int(self._checkleap(ref_year))
            while count+thisyear <= jdays:
                # keep going
                year = year + 1
                count = count + thisyear
                # update thisyear
                if self._checkleap(year):
                    thisyear = 366.0
                else:
                    thisyear = 365.0                        
            jdays = jdays - count # number of days from start of year of interest

        # now need to break down jdays            
        if self._checkleap(year):
            #thisyear = 366.0
            mlist = edays_leap
        else:
            #thisyear = 365.0
            mlist = edays
            
        if jdays >= mlist[-1]:
            month = 12
            day = int(jdays-mlist[-1])+1
        else:
            for mInd in range(11):
                if jdays >=mlist[mInd] and jdays < mlist[mInd+1]:
                    month = mInd+1
                    day = int(jdays - mlist[mInd])+1
                    break

        # initialise utc array to hold result
        utc = [year,month,day,hour,minute,second]
        
        return utc
    
    ##########################################################
    # Function to convert between utc and solls
    def _utc_soll(self,utc):
        
        # Count the Julian day number of the UTC time of interest
        jd_ut = self._utc_julian(utc)
        
        # The conversion function works using Terrestrial Time, which is based on International Atomic Time (IAT)
        # Since UTC has been periodically slowed down via the addition of leap seconds (in order to keep
        # fairly close to apparent solar time as measured by UT1), we need to count the number of leap 
        # seconds added up to the UTC time of interest:
        tai_offset = self._utc_leaps(utc)
        
        # Convert from UTC to TT. The extra 32.184 is to align TT with IAT
        jd_tt = jd_ut + (tai_offset + 32.184) / 86400 #         
        
        # Calculate the time since the J2000 epoch
        j2000 = jd_tt - 2451545.0 # 2451545 - 12:00:00 on 1 Jan 2000    
        
        # Calculate the mars sol date, which is the equivalent of the Julian date for mars
        msd = ( (j2000 - 4.5) / 1.027491252) + 44796.0 - 0.00096
        # 4.5 because it was midnight at the martian prime meridian at midnight on 6th Jan 2000 (it j2000=4.5)
        # 1.027491252 is the ratio of a (mean) mars day to a (mean) earth day
        # 44796 is a convention to keep MSD positive going back to 12:00:00 on Dec 29th 1873
        # 0.00096 is a small adjustment since the midnights were not perfectly aligned
        
        # According to the soll definition, time count starts at midnight on the sol when Ls=0.
        # The MSD of this sol is known (44270), so we subtract this to get the soll value
        soll = msd-44270
        
        return soll

    ##########################################################
    # Function to convert between solls and utc    
    def _soll_utc(self,soll):
        
        # get the msd value of the current soll
        msd = soll+44270
        
        # get the time since the start of the J2000 epoch (in Earth days)
        j2000 = ((msd + 0.00096 - 44796.0) * 1.027491252)+4.5        
        
        # calculate terrestrial time (ephemeris time)
        jd_tt = j2000 + 2451545.0 # 2451545 - 12:00:00 on 1 Jan 2000   
        
        # We cannot invert the leap second function, so we first convert without leap adjustment
        jd_ut_noleap = jd_tt - 32.184/86400 # -tai_offset/86400 is left out
        
        utc_noleap = self._julian_utc(jd_ut_noleap)
        
        # get the number of leap seconds pertinent to this utc
        tai_offset = self._utc_leaps(utc_noleap)   
        
        # there is a chance that the utc in question is within a few seconds of a leap second (before or after)
        # to be safe, compare three different utc possibilities and pick the one that converts closest to the original soll
        # This is rather hacky - perhaps there is a better way? KR
        
        utc_0 = self._julian_utc(jd_ut_noleap-tai_offset/86400)
        utc_p = self._julian_utc(jd_ut_noleap-(tai_offset-1)/86400)
        utc_m = self._julian_utc(jd_ut_noleap-(tai_offset+1)/86400)
        
        err_0 = np.abs(soll-self._utc_soll(utc_0))
        err_p = np.abs(soll-self._utc_soll(utc_p))
        err_m = np.abs(soll-self._utc_soll(utc_m))
        
        if err_0 <= err_p and err_0 <= err_m:
            return utc_0
        
        if err_p <= err_0 and err_p <= err_m:
            return utc_p
        
        if err_m <= err_0 and err_m <= err_p:
            return utc_m
        
    ##########################################################
    # Function to convert between julian date (Earth) and continuous areocentric solar longitude    
    def _julian_lscont(self,jd_ut):
        
        ## Count the Julian day number of the UTC time of interest
        #jd_ut = self._utc_julian(utc)
        
        # Get utc time
        utc = self._julian_utc(jd_ut)
        
        # get the soll and mars year
        soll = self._utc_soll(utc)
        sol,MY = self._soll_sol(soll)
        
        # The conversion function works using Terrestrial Time, which is based on International Atomic Time (IAT)
        # Since UTC has been periodically slowed down via the addition of leap seconds (in order to keep
        # fairly close to apparent solar time as measured by UT1), we need to count the number of leap 
        # seconds added up to the UTC time of interest:
        tai_offset = self._utc_leaps(utc)
        
        # Convert from UTC to TT. The extra 32.184 is to align TT with IAT
        jd_tt = jd_ut + (tai_offset + 32.184) / 86400 #         
        
        # Calculate the time since the J2000 epoch
        j2000 = jd_tt - 2451545.0 # 2451545 - 12:00:00 on 1 Jan 2000   

        # Calculate the mean anomaly of Mars, which is the (time-wise) ratio into the full orbit 
        # multiplied by 360 degrees. The orbit is taken to start at the vernal equinox.
        m = (19.3870 + 0.52402075 * j2000) % 360
        # Note: 19.3870 is the mean  anomaly of mars at the J2000 epoch
        # Note: 0.52402075 = 360/686.972, where 686.972 is the number of earth days in a mars year
        
        # Calculate the angle of the fictitious mean sun, again calculated relative to J2000
        alpha_fms = (270.3863 + 0.52403840 * j2000) % 360
        
#        # Calculate the eccentricity of Mars, with a small correction factor accounting for time variations
#        e = (0.09340 + 2.477E-9 * j2000)
        
        deg2rad = np.pi/180 # to convert from degrees to radians, for using trig functions
        # Calculate the deviations to the orbit due to planetary perturbations
        pbs = \
            0.0071 * np.cos(deg2rad*((0.985626 * j2000 /  2.2353) +  49.409)) + \
            0.0057 * np.cos(deg2rad*((0.985626 * j2000 /  2.7543) + 168.173)) + \
            0.0039 * np.cos(deg2rad*((0.985626 * j2000 /  1.1177) + 191.837)) + \
            0.0037 * np.cos(deg2rad*((0.985626 * j2000 / 15.7866) +  21.736)) + \
            0.0021 * np.cos(deg2rad*((0.985626 * j2000 /  2.1354) +  15.704)) + \
            0.0020 * np.cos(deg2rad*((0.985626 * j2000 /  2.4694) +  95.528)) + \
            0.0018 * np.cos(deg2rad*((0.985626 * j2000 / 32.8493) +  49.095))
        
        
        # Calculate the difference between the actual position of the Sun and the fictitous mean Sun
        # This is the same as the difference between the true anomaly and mean anomaly.
        nu_m = (10.691 + 3.0E-7 * j2000) * np.sin(deg2rad*m) +\
            0.623  * np.sin(deg2rad * 2 * m) + \
            0.050  * np.sin(deg2rad * 3 * m) + \
            0.005  * np.sin(deg2rad *4 * m) + \
            0.0005 * np.sin(deg2rad *5 * m) + pbs

        # Calculate the areocentric solar longitude, ie the actual position of the Sun
        normal_ls = (alpha_fms + nu_m) % 360

        # get start and end solls for this Ls cycle        
        Ls0_firstsoll = np.round(self._year_day*(float(MY)-float(self._offset_year)),self._num_dp)
        Ls0_lastsoll = np.round(self._year_day*(float(MY)-float(self._offset_year)+1.0),self._num_dp) 
        error_soll = 10

        if soll <= Ls0_firstsoll+error_soll: # soll might still be in previous year, so ls may not have wrapped
            if normal_ls <= 180: # ls has already wrapped around
                ls_cont = normal_ls+360*(float(MY)-float(self._offset_year))
            else: # ls has not wrapped round
                ls_cont = normal_ls+360*(float(MY-1)-float(self._offset_year))        
        elif soll > Ls0_firstsoll+error_soll and soll <= Ls0_lastsoll-error_soll:
            # this is the case away from the boundaries, so treat as normal (non-wrapped)
                ls_cont = normal_ls+360*(float(MY)-float(self._offset_year))
        elif soll > Ls0_lastsoll-error_soll: # soll may have jumped to next year, so ls may have wrapped
            if normal_ls <= 180: # ls has already wrapped
                ls_cont = normal_ls+360*(float(MY+1)-float(self._offset_year))        
            else: # ls has not wrapped
                ls_cont = normal_ls+360*(float(MY)-float(self._offset_year))

        return ls_cont


    ##########################################################
    # Function to convert between continuous areocentric solar longitude and julian days
    def _lscont_julian(self,lscont):                
        # As it is not possible to invert the ls calculation function, we have to solve numerically        
                
        # Define the function for testing
        def fun(julian):            
            # convert from julian to continuous ls
            lsj = self._julian_lscont(julian)            
            return lsj-lscont
        
        # convert lscont to ls
        ls,MY = self._lscont_ls(lscont)        
        # Define an initial guess        
        j0 = self._utc_julian(self._soll_utc(self._ls_soll_old(ls,MY)))
        # Perform the numerical rootfinding
        julian = root(fun,j0).x[0]
        
        return julian

    ##########################################################
    # Function to convert between normal and continuous ls representations
    def _ls_lscont(self,ls,MY):
        ls_cont = ls+360*(float(MY)-float(self._offset_year))
        return ls_cont
    
    ##########################################################
    # Function to convert between continuous and normal ls representations
    def _lscont_ls(self,lscont):
        MYshift,ls = divmod(lscont,360)
        MY = MYshift+self._offset_year     
        return ls,MY
    
    ##########################################################
    # Function to convert between julian and ls dates
    def _julian_ls(self,julian):
        # convert to lscont
        lscont = self._julian_lscont(julian)
        # convert to ls
        ls,MY = self._lscont_ls(lscont)
        return ls,MY

    ##########################################################
    # Function to convert between ls and julian dates
    def _ls_julian(self,ls,MY):
        # convert to lscount
        lscont = self._ls_lscont(ls,MY)
        # convert to julian
        julian = self._lscont_julian(lscont)
        return julian
        
    ##########################################################
    # Old version for converting from ls to soll
    # In the newest version, this is only used to provide an initial guess for the numerical solver
    def _ls_soll_old(self,ls,MY):        
        # Error checking
 #       if MY==self._offset_year and ls <0:
 #           raise Exception('ls is out of range! keep ls>0 for MY'+str(self._offset_year))
        if ls <= -360 or ls >= 720:
            raise Exception('ls is out of range! Limit to -360 < ls < 720')    
        # get the start and end sols of this MY
        startsoll,endsoll = self._MY_bds(MY)        
        # get the solls that bound the bulk of this MY
        ls0_left_soll = np.round(self._year_day*(MY-self._offset_year),decimals=1)
        ls0_right_soll = np.round(self._year_day*(MY-self._offset_year+1),decimals=1)    
        if ls > 360: # going past the upper end of the range
            soll = ls0_right_soll + self._ls_sol_old(ls) + self._soll_offset
        elif ls < 0: # going past the lower end of the range
            soll = ls0_left_soll - (self._year_day-self._ls_sol_old(ls)) + self._soll_offset
        else: # staying within the bulk range of this MY
            soll = ls0_left_soll + self._ls_sol_old(ls)  + self._soll_offset
        soll = np.round(soll,decimals=self._num_dp) # round to specified accuracy
        #print('_ls_soll: '+str(soll))
        return soll

    ##########################################################
    # function to convert from ls to sol
    # based on idl routines & python implementation of LJ Ruan  
    # This function is never used directly, but is called through _ls_solMY->_ls_soll_old, then _soll_sol
    # Still required in the new version for providing an initial guess to the numerical solver
    
    def _ls_sol_old(self,ls):
        if (abs(ls) < 1.0e-5):
            if (ls >= 0.0):
                return 0.0
            else:
                return self._year_day
        zteta=ls/self._degrad+self._timeperi
        zx0=2.0*np.arctan(np.tan(0.5*zteta)/np.sqrt((1.+self._e_elips)/(1.-self._e_elips)))
        xref=zx0-self._e_elips*np.sin(zx0)
        zz=xref/(2.*np.pi)
        sol=zz*self._year_day+self._peri_day
        if (sol < 0.0):
            sol=sol+self._year_day
        elif (sol >= self._year_day):
            sol=sol-self._year_day    
        sol = np.round(sol,decimals=self._num_dp) # round to specified accuracy
        #print('_ls_sol: '+str(sol))
        return sol

    ##########################################################
    # Function to convert between UTC and Ls
    def _utc_ls(self,utc):
        julian = self._utc_julian(utc)
        ls,MY = self._julian_ls(julian)
        return ls,MY

    ##########################################################
    # Function to convert between Ls and UTC
    def _ls_utc(self,ls,MY):
        julian = self._ls_julian(ls,MY)
        utc = self._julian_utc(julian)
        return utc

    ##########################################################
    # function to convert from utc to sol
    def _utc_sol(self,utc):        
        # convert to soll
        soll = self._utc_soll(utc)
        # convert to sol
        sol, MY = self._soll_sol(soll)
        return sol, MY

    ##########################################################
    # function to convert from sol to utc
    def _sol_utc(self,sol,MY):
        # convert to soll
        soll = self._sol_soll(sol,MY)
        # convert to utc
        utc = self._soll_utc(soll)
        return utc

    ##########################################################
    # function to convert from utc to index
    def _utc_index(self,utc,**kwargs):        
        # convert to soll
        soll = self._utc_soll(utc)
        # convert to index
        folder, index = self._soll_index(soll,**kwargs)
        return folder, index

    ##########################################################
    # function to convert from index to utc
    def _index_utc(self,folder,index):
        # convert to soll
        soll = self._index_soll(folder,index)
        # convert to utc
        utc = self._soll_utc(soll)
        return utc

    ##########################################################
    # function to count the number of leap seconds that have been added to utc up to a given utc time.
    def _utc_leaps(self,utc):
        '''
        Function to count the number of leap seconds that have been added to utc up to a given utc time.    
        This count is neccessary to translate from utc to terrestrial time (ephemeris time) or international atomic time
        '''

        year = utc[0]; month = utc[1]; #day = utc[2]    
        # Define the set of times where a leap second was added to utc to slow it down
        base_add = 10
        
        # Count the number of leap seconds that have been added at the given utc date
        for ind in range(len(self._leap_list)):       
            lyear = self._leap_list[ind]
            if year > lyear[0]:
                base_add+=1
            else:
                if year == lyear[0] and month > lyear[1]:
                    base_add+=1                
        return base_add


    def _checkleap(self,year):
        # Checks if a given year is a leap year (true if a multiple of 4, but not of 100 (unless simultaneously also a multiple of 400)
        if year%4==0 and (year%100 != 0 or year%400==0):
            return True
        else:
            return False    
        
    def _utc_check(self,utc):
        '''
        Sanity check on a given utc date array.
        '''
        
        if len(utc) != 6:
            raise TypeError('UTC must be a 6-element array of integers.')
        
        year = utc[0]
        month = utc[1]
        day = utc[2]
        hour = utc[3]
        minute = utc[4]
        second = utc[5]

        # make sure all entries are integers
        if year != int(year):
            raise ValueError('UTC year must be an integer.')
        if month != int(month):
            raise ValueError('UTC month must be an integer.')
        if day != int(day):
            raise ValueError('UTC day must be an integer.')
        if hour != int(hour):
            raise ValueError('UTC hour must be an integer.')
        if minute != int(minute):
            raise ValueError('UTC minute must be an integer.')
        if second != int(second):
            raise ValueError('UTC second must be an integer.')

        # year cannot be before 1583 (First full Gregorian calendar year)
        if year < 1583:
            raise ValueError('UTC year must not be less than 1583 (in order to stick to the Gregorian calendar)')

        leap = self._checkleap(year)

        # month checks
        if month <=0 or month >12:
            raise ValueError('UTC month must be between 1 and 12.')
        
        # check that day lies between boundary of given month
        month_lb = np.ones((12,)) # lower bounds
        month_ub = np.array([31,28+int(leap),31,30,31,30,31,31,30,31,30,31]) # upper bounds    
        lb = month_lb[int(month)-1] # lower bound for day of the month
        ub = month_ub[int(month)-1] # upper bound for day of the month    
        if day < lb or day > ub:
            raise ValueError('UTC day must be between '+str(int(lb))+' and '+str(int(ub))+' for UTC month '+str(int(month))+'.')
        if hour < 0 or hour >=24:
            raise ValueError('UTC hour must be between 0 and 23')
        if minute < 0 or minute >=60:
            raise ValueError('UTC minute must be between 0 and 59')
        if second < 0 or second >= 60:
            raise ValueError('UTC second must be between 0 and 59')

           
class time:
    '''
    A class for managing times on Mars. Each instance is meant to represent a fixed time on Mars, 
    and is represented in all the different possible ways (sol, soll, ls, index, utc). 
    The class is meant to ease the manipulation of time calculations in different forms. 
        
    Initialisation examples:
        from tc import time                
        t1 = time('sol',230,34,timespersol=12,blocklength=30)             # sol representation (sol 230 of MY34)        
        t2 = time('soll',2300.3,timespersol=12,blocklength=30)            # soll representation (soll 2300.3)        
        t3 = time('ls',134,24,timespersol=12,blocklength=30)              # ls representation (Ls=134 in MY24)           
        t4 = time('index','d31-61',0,timespersol=12,blocklength=30)       # index representation (folder d31-61, index 0)
        t5 = time('utc',[2021,7,6,12,30,0],timespersol=12,blocklength=30) # utc representation (6th July 2021, 12:30:00 UTC)

    The time objects are callable, and return 2-element tuples (sol, ls, index), a float (soll), or a 6-element array (utc).
    
    Calling examples:
        t1() # returns (230,34)
        t2() # returns 2300.3
        t3() # returns (134,24)
        t4() # returns ('d31-61',15)
        t5() # returns [2021,7,6,12,30,0]
    
    Addition and subtraction operations have been implemented in the 'natural' way based on the time format.
    Note that the operations return new instances of the class, rather than modifying the original object.
    
    Arithmetic examples:
        t11 = t1+10 # sols can be added/subtracted. t11() returns (240.0, 34)       
        t21 = t2 + 250 # solls can be added/subtracted. t21() returns 2550.3
        t31 = t3+300 #  solar longitudes can be added/subtracted. t31() returns (74,25)        
        t41 = t4-1 # indices can be added/subtracted. t41() returns ('d1-31', 359)
        t51 = t5-20 # utc seconds can be added/subtracted. t51() returns [2021, 7, 6, 12, 29, 40]
    
    It is also easy to convert between different time representations of a point using the to() method:
        t12 = t1.to('soll') # converts representation from sol to soll. t12() returns 6916.0
            
    '''
    
    def __init__(self,ttype,val1,val2=None,timespersol=12,blocklength=30):
        '''
        Initialises time point. 
        Inputs:
            ttype: string giving time format. Options: 'sol', 'soll', 'ls', 'index', 'utc'
            
            val1: first argument for time point. 
                sol: time in sols, e.g. 23.3
                soll: time in solls, e.g. 3240.3
                ls: solar longitude, e.g. 358
                index: string giving folder name, e.g. 'd1-31'
                utc: time (on Earth) in Coordinated Universal Time, e.g. [2021,7,6,12,30,0] for 6th July 2021, 12:30:00
            
            val2: second argument for time point (None by default)
                sol: MY of the time point, e.g. 25
                soll: always None (so leave blank)
                ls: MY of the time point, e.g. 25
                index: index of the time point in the gcm output array, e.g. 244
                utc: always None (so leave blank)
            
            timespersol: number of time points in each sol of GCM output. Common values are 12 and 24.
            
            blocklength: the number of sols in each model block. Usually 30.                
        '''
        # set primary representation type and attributes
        if ttype not in ['sol','soll','ls','index','utc']:
            raise ValueError('ttype not recognised: only sol, soll, ls, index, utc are accepted.')
        else:
            self.ttype = ttype            
            self.val1=val1
            self.val2=val2
        
        # attributes for time conversion
        self.timespersol=timespersol
        self.blocklength=blocklength
        # internal time conversion function
        self.tc = tc(timespersol,blocklength)
        
        # update the time values to make them coherent with tc norms        
        self.to(self.ttype,new=False)        

    def __call__(self):
        if self.ttype == 'soll' or self.ttype == 'utc':
            return self.val1
        else:
            return (self.val1,self.val2)        
        
    def to(self,ttype,new=True):
        # function to convert between different time representations internally
        if self.ttype == 'soll':
            # Already in soll representation; do nothing
            soll = self.val1
        else:
            # convert to soll representation
            soll = self.tc.convert(self.ttype,'soll',self.val1,self.val2)
    
        if new is True: # return changes in new object
            if ttype == 'soll':
                return time(ttype,soll,timespersol=self.timespersol,blocklength=self.blocklength)
            elif ttype == 'utc':
                val1 = self.tc.convert('soll',ttype,soll)            
                return time(ttype,val1,timespersol=self.timespersol,blocklength=self.blocklength)
            else:
                val1,val2 = self.tc.convert('soll',ttype,soll)            
                return time(ttype,val1,val2,timespersol=self.timespersol,blocklength=self.blocklength)               
        else: # make changes to the current object
            # update internal ttype
            self.ttype = ttype
    
            # convert to desired representation
            if ttype == 'soll': 
                self.val1,self.val2 = (soll,None)
            elif ttype == 'utc':
                self.val1,self.val2 = (self.tc.convert('soll',ttype,soll),None)
            else:
                self.val1,self.val2 = self.tc.convert('soll',ttype,soll)
            return self
            
    def __add__(self,val):
        # function to perform addition on the primary time representation
        if self.ttype == 'sol':
            # convert to soll
            soll = self.tc.convert('sol','soll',self.val1,self.val2)            
            # convert back with increment
            val1,val2 = self.tc.convert('soll','sol',soll+val)        
        elif self.ttype == 'soll':
            val1=self.val1+val
            val2=None
        elif self.ttype == 'ls':
            yearinc,lsinc = divmod(self.val1+val,360) # returns the year increment and new ls value
            val1,val2 = (lsinc,int(self.val2+yearinc))
        elif self.ttype == 'index':
            # convert to soll
            soll = self.tc.convert('index','soll',self.val1,self.val2)
            # convert back with increment
            (val1,val2) = self.tc.convert('soll','index',soll+val/self.timespersol)        
        elif self.ttype == 'utc':
            # convert to julian
            julian = self.tc._utc_julian(self.val1)
            # convert back with increment
            val1= self.tc._julian_utc(julian + val/86400)
            val2=None
        return time(self.ttype,val1,val2,timespersol=self.timespersol,blocklength=self.blocklength)
    
    def __sub__(self,val):
        return self+(-1*val)
    
    def info(self):
        '''
        Prints information about current time object.
        '''
        self.tc.soll_info(self.to('soll')())

    
##########################################################################################
# To run the tc conversion from the command line
# KR Jul 2021
#import sys; sys.path.append(r'/home/stem/kr6978/Documents/utils')
#from tc import tc
import argparse
from datetime import datetime

if __name__ == '__main__':

    # record current time
    now = datetime.now()

    # initialise the parser
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
        description='''
    Function to convert between five different Martian time formats.
    Format types:
        'ls'   : Areocentric solar longitude for a given Mars Year.   Arguments: ls MY
        'sol'  : Number of sols since the start of a given Mars Year. Arguments: sol MY
        'soll' : Number of sols since the start of MY24.              Arguments: soll
        'index': Folder name and time index in the OU MGCM.           Arguments: fol index
        'utc'  : Earth time in Univeral Coordinated Time.             Arguments: dd/mm/yyyy hh:mm:ss''',
        epilog = '''
    Author: Kylash Rajendran 
    kylash.rajendran@open.ac.uk''')

    p.add_argument('-i',
                '--input',
                type=str,
                nargs='*',
                action='store',
                help='''Input time format and values.
    Example: All configurations below refer to the same time point, ls=180 degrees of Mars Year 34:
        -i ls 180.0 34
        -i sol 372.04 34
        -i soll 7058.04
        -i index d7051-7081 96
        -i utc 22/05/2018 14:46:55''')

    p.add_argument('-o',
                '--output',
                type=str,
                help='''Choose a specific output time format.
    Options: ls, sol, soll, index, utc.''')

    p.add_argument('-t',
                '--timespersol',
                type=int,
                default=12,
                action='store',
                help='The number of model output times per sol, default=12.')

    p.add_argument('-b',
                '--blocklength',
                type=int,
                default=30,
                action='store',
                help='The number of sols in a model block, default=30.')

    p.add_argument('-c',
                '--clean',
                action='store_true',
                help='Generate output without decorations.')

    args = p.parse_args()

    # Initialise tc object
    tcx = tc(args.timespersol,args.blocklength)
    # convert times to soll for reference
    if args.input == None: # use current time as default        
        headStr='Current Mars time:'
        utc = [now.year,now.month,now.day,now.hour,now.minute,now.second]    
        soll = tcx.convert('utc','soll',utc)
    else: # input times given
        headStr='Mars time:'
        if args.input[0] == 'soll':
            if len(args.input) != 2:
                raise ValueError('tc: incorrect parameters. For -i soll, 1 additional parameter is required: soll.')
            soll = float(args.input[1])        
        elif args.input[0] == 'ls':
            if len(args.input) != 3:
                raise ValueError('tc: incorrect parameters. For -i ls, 2 additional parameters are required: ls MY.')
            ls = float(args.input[1]); MY = int(args.input[2])
            soll = tcx.convert('ls','soll',ls,MY)
        elif args.input[0] == 'sol':
            if len(args.input) != 3:
                raise ValueError('tc: incorrect parameters. For -i sol, 2 additional parameters are required: sol MY.')
            sol = float(args.input[1]); MY = int(args.input[2])
            soll = tcx.convert('sol','soll',sol,MY)
        elif args.input[0] == 'index':
            if len(args.input) != 3:
                raise ValueError('tc: incorrect parameters. For -i index, 2 additional parameters are required: folder index.')
            fol = args.input[1]; index = int(args.input[2])
            soll = tcx.convert('index','soll',fol,index)
        elif args.input[0] == 'utc':
            if len(args.input) != 3:                
                if len(args.input) == 2:
                    warnmsg = 'tc: Expected two additional parameters for -i utc, but received only 1. Assuming input parameter is date only and calculating for midnight UTC.'
                    warnings.warn(warnmsg)
                    dte = args.input[1];
                    sloc = [pos for pos, char in enumerate(dte) if char == '/']
                    if len(sloc) != 2:
                        raise ValueError('utc input date format must be dd/mm/yyyy (use leading zeros where necessary)')
                    utc = [int(dte[(sloc[1]+1):]),int(dte[(sloc[0]+1):sloc[1]]),int(dte[0:sloc[0]]),0,0,0]
                    
                    soll = tcx.convert('utc','soll',utc)
                else:
                    raise ValueError('tc: incorrect parameters. For -i utc, 2 additional parameters are required: dd/mm/yyyy hh:mm:ss.')        
            else:    
                dte = args.input[1]; tim = args.input[2]
                sloc = [pos for pos, char in enumerate(dte) if char == '/']
                cloc = [pos for pos, char in enumerate(tim) if char == ':']

                if len(sloc) != 2 or len(cloc) !=2:
                    raise ValueError('utc input format is -i utc dd/mm/yyyy hh:mm:ss (use leading zeros where necessary)')
                utc = [int(dte[(sloc[1]+1):]),int(dte[(sloc[0]+1):sloc[1]]),int(dte[0:sloc[0]]),int(tim[0:cloc[0]]),int(tim[(cloc[0]+1):cloc[1]]),int(tim[(cloc[1]+1):])]
                # 7 parameter version:
        #        if len(args.input) != 7:
        #            raise ValueError('tc: incorrect parameters. For -i utc, 6 additional parameters are required: year month day hour minute second.')
        #        utc = [int(args.input[i]) for i in [1,2,3,4,5,6]]
                soll = tcx.convert('utc','soll',utc)
        else: # format not recognised
            raise ValueError('input type '+str(args.input[0])+' not recognised.')
    #Generate set of strings for printing to terminal:
    outlist = []
    if args.clean is False:
        outlist.append(headStr)
    if args.output == None or args.output == 'ls':             
        ls,MYls = tcx.convert('soll','ls',soll)
        if args.clean is False:
            outstr = 'ls: '+"{:.2f}".format(ls)+', MY'+str(int(MYls))
        else:
            outstr = 'ls,MY: '+"{:.2f}".format(ls)+','+str(int(MYls))
        outlist.append(outstr)
    if args.output == None or args.output == 'sol':            
        sol,MYsol = tcx.convert('soll','sol',soll)    
        if args.clean is False:
            outstr = 'sol: '+"{:.2f}".format(sol)+', MY'+str(int(MYsol))
        else:
            outstr = 'sol,MY: '+"{:.2f}".format(sol)+','+str(int(MYsol))
        outlist.append(outstr)
    if args.output == None or args.output == 'soll':        
        if args.clean is False:
            outstr = 'soll: '+"{:.2f}".format(soll)+' sols since MY24 sol 0.0'
        else:
            outstr = 'soll: '+"{:.2f}".format(soll)
        outlist.append(outstr)
    if args.output == None or args.output == 'index':    
        fol,index = tcx.convert('soll','index',soll)
        if args.clean is False:
            outstr = 'index: '+fol+', index: '+str(index)
        else:
            outstr = 'folder,index: '+fol+','+str(index)
        outlist.append(outstr)
    if args.output == None or args.output == 'utc':    
        utc = tcx.convert ('soll','utc',soll)
        if args.clean is False:
            outstr = 'UTC time: '+str(utc[2]).zfill(2)+r'/'+str(utc[1]).zfill(2)+r'/'+str(utc[0]).zfill(4)+', '+str(utc[3]).zfill(2)+':'+str(utc[4]).zfill(2)+':'+str(utc[5]).zfill(2)
        else:
            outstr = 'utc: '+str(utc[2]).zfill(2)+r'/'+str(utc[1]).zfill(2)+r'/'+str(utc[0]).zfill(4)+','+str(utc[3]).zfill(2)+':'+str(utc[4]).zfill(2)+':'+str(utc[5]).zfill(2)
        outlist.append(outstr)
    if args.clean is True:
        # print output without bounding box
        for s in outlist:
            print(s)
    else: 
        # Configure a bounding box and print the output
        osym = '*'
        maxlen = max([len(s) for s in outlist])
        print(osym*(maxlen+4))
        for s in outlist:
            newstr = osym+' '+s+' '*(maxlen-len(s)+1)+osym
            print(newstr)
        print(osym*(maxlen+4))
 
 
    
