from xml.etree.ElementPath import find
import numpy as np
import math

''' astroConstants (from Horizon: 16.09.2022)
This functions returns the astrodynamic-related physical constants * to be updated *

List of identifiers:
    1   Universal gravity constant (G) [km^3/(kg*s^2)]
    2   Astronomical Unit (AU) [km]
    3   Speed of light in the vacuum [km/s]
    4   Standard free fall (the acceleration due to gravity on the Earth's surface at sea level) [km/s**2]
    5   Solar constant = energy flux per unit time per unit area at 1 AU (Earth) [W/m**2 = kg/s**3]

    * Space for other constants *

    Planetary constant [km^3/s^2]: (mu = mass * G)
    10  Sun
    11  Mercury 
    12  Venus
    13  Earth
    14  Mars system
    15  Jupiter system
    16  Saturn system
    17  Uranus system
    18  Neptune system
    19  Moon

    Mean radius [km]:
    20  Sun
    21  Mercury
    22  Venus
    23  Earth
    24  Mars
    25  Jupiter
    26  Saturn
    27  Uranus
    28  Neptune
    29  Moon

    Second zonal harmonics J2 [-]:
    31  Mercury
    32  Venus
    33  Earth
    34  Mars
    35  Jupiter
    36  Saturn
    37  Uranus
    38  Neptune
    39  Moon
'''
def astroConstants(input):

    if (input == 1):
        return 6.67430e-20
    elif (input == 2):
        return 149597870.7
    elif (input == 3):
        return 299792.458
    elif (input == 4):
        return 9.80665*1e-3
    elif (input == 5):
        return 1367

    elif (input == 10):
        return 1.327124400181e11
    elif (input == 11):
        return 22031.868551
    elif (input == 12):
        return 324858.592000
    elif (input == 13):
        return 398600.435507
    elif (input == 14):
        return 42828.375816
    elif (input == 15):
        return 126712764.100000
    elif (input == 16):
        return 37940584.841800
    elif (input == 17):
        return 5794556.400000
    elif (input == 18):
        return 6836527.100580
    elif (input == 19):
        return 4902.800118

    elif (input == 20):
        return 6.9551e5
    elif (input == 21):
        return 2439.4 
    elif (input == 22):
        return 6051.8 
    elif (input == 23):
        return 6371.0084
    elif (input == 24):
        return 3389.50
    elif (input == 25):
        return 69911 
    elif (input == 26):
        return 58232
    elif (input == 27):
        return 25362
    elif (input == 28):
        return 24622
    elif (input == 29):
        return 1737.4

    elif (input == 31):
        return 60e-6
    elif (input == 32):
        return 4.458e-6 
    elif (input == 33):
        return 1.08263e-3
    elif (input == 34):
        return 1.96045e-3
    elif (input == 35):
        return 14.736e-3
    elif (input == 36):
        return 16.298e-3
    elif (input == 37):
        return 3.34343e-3
    elif (input == 38):
        return 3.411e-3
    elif (input == 39):
        return 202.7e-6
    
    else:
        print('Number not yet saved!')


''' atmosphereDensityEarth (from Horizon: 16.09.2022)
Evaluation of the atmospheric density of Earth given a specific positition reference
Input:
- position : Cartesian object, position of the spacecraft with respect to the center of Earth
Output:
- atmospheric density [kg/km**3]
'''
def atmosphereDensityEarth(position):

    radius_earth = astroConstants(23) # [km]
    altitude = position.normalise() - radius_earth
    
    if (altitude < 0):
        print('The atmospheric density cannot be evaluated since the altitude is negative!')
        density = None
    elif (altitude > 1000): # [km]
        density = 0
    else:
        reference_altitude = np.linspace(0, 1000, 201) # [km]
        reference_density = np.array( [1.2250E+00, 7.3643E-01, 4.1351E-01, 1.9476E-01, 8.8910E-02, 4.0084E-02, 1.8410E-02, 8.4634E-03, 
        3.9957E-03, 1.9663E-03, 1.0269E-03, 5.6810E-04, 3.0968E-04, 1.6321E-04, 8.2829E-05, 3.9921E-05, 1.8458E-05, 8.2195E-06, 
        3.4400E-06, 1.3873E-06, 5.6044E-07, 2.3325E-07, 9.6734E-08, 4.2794E-08, 2.2199E-08, 1.2918E-08, 8.1494E-09, 5.4647E-09, 
        3.8313E-09, 2.7805E-09, 2.0752E-09, 1.5848E-09, 1.2336E-09, 9.7526E-10, 7.8155E-10, 6.3382E-10, 5.1940E-10, 4.2952E-10, 
        3.5807E-10, 3.0064E-10, 2.5407E-10, 2.1596E-10, 1.8456E-10, 1.5849E-10, 1.3671E-10, 1.1839E-10, 1.0290E-10, 8.9757E-11, 
        7.8550E-11, 6.8954E-11, 6.0706E-11, 5.3587E-11, 4.7420E-11, 4.2058E-11, 3.7382E-11, 3.3294E-11, 2.9710E-11, 2.6560E-11, 
        2.3783E-11, 2.1331E-11, 1.9159E-11, 1.7232E-11, 1.5519E-11, 1.3994E-11, 1.2634E-11, 1.1419E-11, 1.0333E-11, 9.3607E-12, 
        8.4886E-12, 7.7054E-12, 7.0011E-12, 6.3670E-12, 5.7954E-12, 5.2795E-12, 4.8132E-12, 4.3914E-12, 4.0093E-12, 3.6629E-12, 
        3.3484E-12, 3.0627E-12, 2.8028E-12, 2.5662E-12, 2.3507E-12, 2.1543E-12, 1.9752E-12, 1.8118E-12, 1.6626E-12, 1.5265E-12, 
        1.4020E-12, 1.2883E-12, 1.1843E-12, 1.0891E-12, 1.0020E-12, 9.2222E-13, 8.4913E-13, 7.8214E-13, 7.2070E-13, 6.6434E-13, 
        6.1261E-13, 5.6511E-13, 5.2148E-13, 4.8139E-13, 4.4454E-13, 4.1065E-13, 3.7949E-13, 3.5083E-13, 3.2446E-13, 3.0019E-13, 
        2.7785E-13, 2.5727E-13, 2.3832E-13, 2.2086E-13, 2.0477E-13, 1.8993E-13, 1.7625E-13, 1.6363E-13, 1.5199E-13, 1.4124E-13, 
        1.3131E-13, 1.2214E-13, 1.1367E-13, 1.0584E-13, 9.8597E-14, 9.1903E-14, 8.5710E-14, 7.9981E-14, 7.4678E-14, 6.9770E-14, 
        6.5225E-14, 6.1015E-14, 5.7114E-14, 5.3499E-14, 5.0147E-14, 4.7038E-14, 4.4154E-14, 4.1478E-14, 3.8993E-14, 3.6686E-14, 
        3.4542E-14, 3.2550E-14, 3.0698E-14, 2.8976E-14, 2.7374E-14, 2.5882E-14, 2.4491E-14, 2.3196E-14, 2.1987E-14, 2.0858E-14, 
        1.9805E-14, 1.8820E-14, 1.7900E-14, 1.7039E-14, 1.6233E-14, 1.5478E-14, 1.4771E-14, 1.4108E-14, 1.3487E-14, 1.2903E-14, 
        1.2356E-14, 1.1841E-14, 1.1358E-14, 1.0904E-14, 1.0477E-14, 1.0074E-14, 9.6947E-15, 9.3370E-15, 8.9992E-15, 8.6801E-15, 
        8.3783E-15, 8.0928E-15, 7.8223E-15, 7.5659E-15, 7.3227E-15, 7.0917E-15, 6.8723E-15, 6.6635E-15, 6.4649E-15, 6.2757E-15, 
        6.0952E-15, 5.9231E-15, 5.7587E-15, 5.6016E-15, 5.4514E-15, 5.3075E-15, 5.1698E-15, 5.0377E-15, 4.9111E-15, 4.7895E-15, 
        4.6728E-15, 4.5605E-15, 4.4525E-15, 4.3486E-15, 4.2484E-15, 4.1519E-15, 4.0587E-15, 3.9687E-15, 3.8818E-15, 3.7977E-15, 
        3.7163E-15, 3.6375E-15, 3.6375E-15] ) # [kg/m**3]

        index = np.array(np.where(reference_altitude <= altitude))
        
        density2 = reference_density[index.max() + 1]
        density1 = reference_density[index.max()]
        altitude2 = reference_altitude[index.max() + 1]
        altitude1 = reference_altitude[index.max()]

        density = ((density1 - density2)/(altitude1 - altitude2) * altitude + (density2*altitude1 - density1*altitude2)/(altitude1 - altitude2))*1e9 # [kg/km**3]

    return density # [kg/km**3]


''' date2J2000
Permits to evaluate the Julian day number given the input date 
(valid from year 1901 until year 2099)
'''
def date2J2000(year, month, day, hrs, min, sec):

    J0 = 367*year - math.floor(7/4*(year + math.floor((month + 9)/12))) + math.floor(275/9*month) + day + 1721013.5 # [days]
    UT = hrs + min/60 + sec/3600 # [hrs]
    JD = J0 + UT/24 # [days]

    return JD


def fraction2month_day(year, fraction):

    number_days = np.floor(fraction)

    m = np.array([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]) # days of the months
    if (np.abs(2024 - year) % 4 == 0): # leap year
        m[1] = 29

    for cont in range(1, m.size):
        m[cont] = m[cont] + m[cont-1]
    
    index = np.array(np.where(m < number_days))

    month = index.max() + 1
    day = number_days - m[(month - 1)]

    hrs = np.floor( (fraction - number_days)*24 )
    min = np.floor( ((fraction - number_days)*24 - hrs)*60 )
    sec = (((fraction - number_days)*24 - hrs)*60 - min)*60

    return [year, month, day, hrs, min, sec]


    