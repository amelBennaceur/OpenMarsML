CSV file containing data measured by the NASA Insight lander on the Martian surface, with corresponding assimilation data from the OpenMARS reanalysis dataset.
Observation data has been downsampled to hourly output (from 1-20Hz original measurement frequency)

Contact: kylash.rajendran@open.ac.uk


# Description of columns:

* Time: Observation time in UTC

* Sol: Number of martian days ('sols') since the start of Mars Year 24. (Note: On average, each martian year is 668.6 sols long) 

* Ls: The solar longitude of the observation time, in degrees. Solar longitude ranges from 0 to 360 and represents the seasonal progression of Mars along its orbit. (Northern) Spring equinox occurs at Ls=0, summer solstice at Ls=90, autumn equinox at Ls=180 and winter at Ls=270.

* LT: Local time of the observation, in 24-hour format. (Range: 0-23)

* Psurf_assim: Surface pressure in Pascals at the InSight landing site, as simulated in the OpenMARS reanalysis. 

* u_assim: Eastward horizontal wind speed (m/s) ~5 m above the ground at the InSight landing site, as simulated in the OpenMARS reanalysis. 

* v_assim: Northward horizontal wind speed (m/s) ~5m above the ground at the InSight landing site, as simulated in the OpenMARS reanalysis.

* dust_assim: column dust optical depth (dimensionless) at the InSight landing site, as simulated in the OpenMARS reanalysis.

* temp_assim: Air temperature (K) at the InSight landing site, ~5m above the ground as simulated in the OpenMARS reanalysis. 5m is the approximate height of the lowest model level.

* Psurf_obs: Surface pressure in Pascals at the InSight landing site, as measured by the InSight pressure sensor.

* temp_obs: Air temperature (K) at the InSight landing site, measured by the InSight TWINS sensor 1.4m above the ground.

* u_obs: Eastward horizontal wind speed (m/s) at the InSight landing site, measured by the InSight TWINS sensor 1.4m above the ground.

* v_obs: Northward horizontal wind speed (m/s) at the InSight landing site, measured by the InSight TWINS sensor 1.4m above the ground.
