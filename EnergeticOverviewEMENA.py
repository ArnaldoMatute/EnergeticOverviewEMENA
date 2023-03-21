# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 19:02:53 2023

@author: USUARIO
"""

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import geopandas as gpd
import pandas as pd
import numpy as np
from cartopy.io import shapereader
import cartopy.crs as ccrs
import cartopy
from matplotlib.offsetbox import AnchoredText

plt.rcParams['savefig.dpi'] = 500


## Function to plot piecharts anywhere in a scatterplot with desired radius
## This function has been taken from 
## https://stackoverflow.com/questions/56337732/how-to-plot-scatter-pie-chart-using-matplotlib
## Thank you Quang Hoang!

def draw_pie(dist, 
             xpos, 
             ypos, 
             size, colorss, 
             ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10,8))

    # for incremental pie slices
    cumsum = np.cumsum(dist)
    cumsum = cumsum/ cumsum[-1]
    pie = [0] + cumsum.tolist()
    k=0
    for r1, r2 in zip(pie[:-1], pie[1:]):
        angles = np.linspace(2 * np.pi * r1, 2 * np.pi * r2)
        x = [0] + np.cos(angles).tolist()
        y = [0] + np.sin(angles).tolist()

        xy = np.column_stack([x, y])

        ax.scatter([xpos], [ypos], marker=xy, s=size,
                   edgecolor = "black", linewidth = 1, color = colorss[k],
                   transform = ccrs.PlateCarree(), alpha = 1)
        k+=1

    return ax

#Calculates Energy consumption ratio per GDP in GWh / Million USD of 2015
def quick_calc_rel(x, y): 
    if y != 0:
        y = y*1e-6
        r = x*1e-3/y
    else:
        r = np.nan    
    return r

## Loading files
## The used version of dataset "World Energy Consumption.csv" has not GDP data for the year
## under scope (2019). Hence, it was separately downloaded directly from "Our World in Data"
## as the 'gross-domestic-product.csv' file.
Energy=pd.read_csv('World Energy Consumption.csv') # Energy Dataset
GDP=pd.read_csv('gross-domestic-product.csv')      # GDP dataset

# Parameters for retrieving geographical info from Cartopy 
resolution = '10m'
category = 'cultural'
name = 'admin_0_countries'

shpfilename = shapereader.natural_earth(resolution, category, name)

reader = shapereader.Reader(shpfilename)
countries = reader.records()

df = gpd.read_file(shpfilename)

# Just keeping countries of Europe, Middle East and North Africa (EMENA)
ECA = df.loc[df['REGION_WB'] == 'Europe & Central Asia',['ADMIN','geometry']]
NAME = df.loc[df['REGION_WB'] == 'Middle East & North Africa',['ADMIN','geometry']]
EMENA = pd.concat([ECA, NAME])

## Excluding regions without data available in Energy dataset or out of the scope of
## this analysis
to_excl = ["Greenland", "Dhekelia Sovereign Base Area",
           "Uzbekistan", "Kazakhstan", "Tajikistan", "Kyrgyzstan",
           "Monaco", "Turkmenistan", "Gibraltar", "Vatican", "Northern Cyprus",
           "Cyprus No Mans Area", "Baykonur Cosmodrome", "Akrotiri Sovereign Base Area",
           "Jersey", "Guernsey", "Isle of Man", "Aland", "Faroe Islands", "Djibouti",
           "Iran", "Qatar", "Kuwait", "Yemen", "Bir Tawil", "Bahrain",
           "Georgia", "Armenia", "Azerbaijan", "Liechtenstein", "San Marino",
           "Andorra", "Oman", "Western Sahara", "Iraq", "United Arab Emirates"]

for c in to_excl:    
    EMENA = EMENA.loc[EMENA['ADMIN'] != c,:]
    
## Organizing EMENA Dataset
EMENA = EMENA.rename(columns = {"ADMIN" : "country"})

EMENA = EMENA.set_index("country")    

## Keeping all names the same in all datasets
EMENA = EMENA.rename(index = {"Republic of Serbia" : "Serbia",
                              "Bosnia and Herzegovina" : "Bosnia and Herz."})

EMENA.sort_index(axis = 0, ascending = True, inplace = True)

EmenaCs = EMENA.index.to_list() # List of countries in scope


## List of Tech in scope and their colors for the Ad Hoc Legend

Techs = ["biofuel", "coal", "gas", "hydro","nuclear", "oil", "solar", "wind", "Other"]    
lcolor = ["orange", "black", "gray", "cyan", "green", "yellow", "red", "blue", "white"]

## Dictionary that links technology to color
ColsToTech = {t: c for t, c in zip(Techs,lcolor)}

## Selecting just the columns under scope in the Energy dataset
scope_cols = ["country", "per_capita_electricity" ,"population", "gdp"]

for i in Energy.columns:
    for t in Techs:
        if "elec" in i and "share" in i and t in i:
            scope_cols.append(i)

## Filtering just 2019 data from Energy and organizing
Energy2019 = Energy.loc[Energy["year"] == 2019, scope_cols]

Energy2019.insert(len(Energy2019.columns),"Other", value = 100 - 
                               Energy2019.loc[:,scope_cols[4:]].sum(axis = 1))

Energy2019 = Energy2019.set_index("country")

Energy2019.sort_index(axis = 0, ascending = True, inplace = True)

## Filtering just 2019 data from GDP and organizing
GDP2019 = GDP.loc[GDP["Year"] == 2019, ['Entity',"GDP (constant 2015 US$)"]]

GDP2019 = GDP2019.rename(columns = {"Entity" : "country",
                                    "GDP (constant 2015 US$)" : "GDP_USD(2015)"})

GDP2019 = GDP2019.set_index("country")  

## Completing GDP data for 2019 in Energy dataset since originally this info was missing
## Just countries in scope have been considered

Energy2019 = Energy2019.rename(index = {"Bosnia and Herzegovina" : "Bosnia and Herz."})

GDP2019 = GDP2019.rename(index = {"Bosnia and Herzegovina" : "Bosnia and Herz."})

## Transfering GDP data from GDP dataset to Energy2019 datset 
for c in EmenaCs:
    Energy2019.at[c, "gdp"] = GDP2019.loc[c, "GDP_USD(2015)"]
    
## Inserting two new columns for total consumption and energy per GDP    

Energy2019.insert(len(Energy2019.columns), "total_consumption", value = np.nan)
Energy2019.insert(len(Energy2019.columns), "energy_per_gdp", value = np.nan)

## Calculating total electricity consumption for each country
Energy2019["total_consumption"] = Energy2019.apply(
    lambda x: x.per_capita_electricity*x.population, axis = 1)

## Calculatiing consumption/GDP ratio for countries in scope
Energy2019["energy_per_gdp"] = Energy2019.apply(
    lambda x: quick_calc_rel(x.total_consumption, x.gdp), axis = 1)


## Min and max of energy/gdp ratio

Emin = np.min(Energy2019["energy_per_gdp"])
Emax = np.max(Energy2019["energy_per_gdp"])

norm = plt.Normalize(Emin, Emax)

##### LAYOUT CONSTRUCTION ########

## Style of boxes
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)

## Define all axes objects
f = plt.figure(figsize = (12,12))
ax_tech = f.add_axes([0,0.79,0.55,0.08])
ax = f.add_axes([0,0,0.8,1], projection = ccrs.PlateCarree())
ax_cir = f.add_axes([0.6,0.79,0.2,0.08])
bax = f.add_axes([0.84, 0.225, 0.02, 0.65])

## Positions for Ad Hoc legend for technologies
xp = [0.05, 0.05, 0.25, 0.25, 0.45, 0.45, 0.65, 0.65, 0.85]
yp = [0.4, 0.7, 0.4, 0.7, 0.4, 0.7, 0.4, 0.7, 0.7]


## ax_tech is the axes Ad Hoc legend for technologies 
## Set limits
ax_tech.set_ylim(0.2, 0.9)
ax_tech.set_xlim(0, 1)

## Ad hoc legend for technologies
for x,y,c in zip(xp,yp,ColsToTech.items()):
    ax_tech.scatter(x, y, c = c[1], edgecolors = "black", s = 180)
    ax_tech.text(x = x + 0.08, y = y,
                    s = c[0], ha="center", va="center", size=12, bbox = bbox_props)

## Title for the axes which will be an Ad Hoc legend
ax_tech.set_title("Technologies")

## Setting extent for desired geographic areas.
## Under EMENA all region under scope will be displayed.
## For the sake of better visualization, there are extent for the Balkans and Middle East

# Uncomment the desired extent, and related variables
# Comment other when not desired to be displayed
####################################### EMENA #########################################
ax.set_extent([-25, 39, 27, 72], crs=ccrs.PlateCarree())
coe = 0.08

lab1 = "7GWh"
lab2 = "25GWh"
lab3 = "60GWh"


####################################### Balkans #######################################
#ax.set_extent([10, 30, 36, 50], crs=ccrs.PlateCarree())
#coe = 0.18
#text2 = AnchoredText('No full data available for Kosovo and North Macedonia',
#                        loc = "lower left", prop={'size': 12}, frameon=True)
#ax.add_artist(text2)

#lab1 = "3GWh"
#lab2 = "8GWh"
#lab3 = "20GWh"

################################## Middle East #########################################
#ax.set_extent([21, 38, 28, 40], crs=ccrs.PlateCarree())
#coe = 0.25
#text3 = AnchoredText('No data available for Northern Cyprus',
#                        loc = "lower left", prop={'size': 12}, frameon=True)
#ax.add_artist(text3)

#lab1 = "3GWh"
#lab2 = "8GWh"
#lab3 = "20GWh"
#########################################################################################

## Ad Hoc legend for production
## Due to the non-linearity of circular markers relation to their radii 
## it was preferred to code this part step by step

## ax_cir is the axes for legend related to the size of the piechart circle
## Set limits
ax_cir.set_ylim(-0.2, 1.6) 
ax_cir.set_xlim(-0.4, 1.4)

## Set circles    
ax_cir.scatter(0.25,0.65, s = 4500, c = "Lightgray", edgecolor = "Black")
ax_cir.scatter(0.25,0.375, s = 2000, c = "Lightgray", edgecolor = "Black")
ax_cir.scatter(0.25,0.15, s = 560, c = "Lightgray", edgecolor = "Black")

## Set lines
ax_cir.plot([0.2,0.75],[0.45,0.45], c="black", linewidth = 1)
ax_cir.plot([0.2,0.75],[0.95,0.95], c="black", linewidth = 1)
ax_cir.plot([0.2,0.75],[1.5,1.5], c="black", linewidth = 1)

## Set textboxes
ax_cir.text(x = 0.95, y = 0.2, s = lab1, ha="center", va="center",
            size = 8, bbox = bbox_props)

ax_cir.text(x = 0.95, y = 0.7, s = lab2, ha="center", va="center",
                size = 8, bbox = bbox_props)

ax_cir.text(x = 0.95, y = 1.3, s = lab3, ha="center", va="center",
                size = 8, bbox = bbox_props)

## Title for the axes which will be an Ad Hoc legend
ax_cir.set_title("Per capita total consumption")

## Colormap for Consumption / GDP ratio
cmap = get_cmap('summer')

## for all countries under scope
for p in EMENA.index:
    
    ## Select color regarding the ratio
    gradient = cmap(norm(Energy2019.loc[p,"energy_per_gdp"]))
    ## Add geometry of the country
    ax.add_geometries(EMENA.loc[p,"geometry"], crs=ccrs.PlateCarree(),
                      zorder = 0, facecolor = gradient)

## Add borders, coastlines and oceans    
ax.add_feature(cartopy.feature.COASTLINE, zorder = 0)
ax.add_feature(cartopy.feature.BORDERS, linewidth = 1, edgecolor = "black", zorder = 0)
ax.add_feature(cartopy.feature.OCEAN, facecolor = "lightblue")

## Add textbox for citing the soures
text = AnchoredText('Source: "Our World in Data", Hannah Ritchie, Max Roser and Edouard Mathieu. License: Creative Commons BY. ',
                        loc = "upper center", prop={'size': 12}, frameon=True)
ax.add_artist(text)

## Add Other category for countries which use technologies beyond those in scope
scope_cols.append("Other")

## For all contries under scope
for country in countries:

    for c in EMENA.index:

        if country.attributes['NAME'] == c:
            
## Get the centroid of each country   
            x = country.geometry.centroid.x        
            y = country.geometry.centroid.y

## Calculate a list with all shares of consumption regarding generation technology            
            A = np.ravel(Energy2019.loc[c,scope_cols[4:]].to_numpy())
            
            MkrSize = "per_capita_electricity"

## Some centroids provided by cartopy need to be slightly moved for better visualization
## among scattered piecharts circles and selected extent.
## They had to be manually moved since the replacement is different for all of them 
             
            if c == "France":
                draw_pie(A, x + 5, y + 5, coe*Energy2019.loc[c, MkrSize], colorss = lcolor, ax = ax)
            elif c == "Algeria":
                    draw_pie(A, x, y + 5, coe*Energy2019.loc[c, MkrSize], colorss = lcolor, ax = ax)
            elif c == "Russia":
                    draw_pie(A, x - 63, y - 5, coe*Energy2019.loc[c, MkrSize], colorss = lcolor, ax = ax)
            elif c == "Norway":
                    draw_pie(A, x - 8, y - 7, coe*Energy2019.loc[c, MkrSize], colorss = lcolor, ax = ax)
            elif c == "Morocco":
                     draw_pie(A, x + 2, y + 2, coe*Energy2019.loc[c, MkrSize], colorss = lcolor, ax = ax)
            elif c == "Libya":
                     draw_pie(A, x, y + 2, coe*Energy2019.loc[c, MkrSize], colorss = lcolor, ax = ax)
            elif c == "Egypt":
                     draw_pie(A, x - 1, y + 3, coe*Energy2019.loc[c, MkrSize], colorss = lcolor, ax = ax)
            elif c == "Malta":
                     draw_pie(A, x + 1.5, y, coe*Energy2019.loc[c, MkrSize], colorss = lcolor, ax = ax)
            elif c == "Cyprus":
                     draw_pie(A, x - 2, y, coe*Energy2019.loc[c, MkrSize], colorss = lcolor, ax = ax)
            elif c == "Saudi Arabia":
                     draw_pie(A, x - 7.5, y + 5, coe*Energy2019.loc[c, MkrSize], colorss = lcolor, ax = ax)
            elif c == "Israel":
                     draw_pie(A, x - 1, y + 1, coe*Energy2019.loc[c, MkrSize], colorss = lcolor, ax = ax)
            elif c == "Lebanon":
                     draw_pie(A, x - 0.6, y + 0.35, coe*Energy2019.loc[c, MkrSize], colorss = lcolor, ax = ax)
            elif c == "Palestine":
                    draw_pie(A, x, y - 0.3, coe*Energy2019.loc[c, MkrSize], colorss = lcolor, ax = ax)
            elif c == "Jordan":
                    draw_pie(A, x, y + 0.5, coe*Energy2019.loc[c, MkrSize], colorss = lcolor, ax = ax)
            elif c == "Turkey":
                   draw_pie(A, x - 3, y - 1, coe*Energy2019.loc[c, MkrSize], colorss = lcolor, ax = ax)
            elif c == "Iceland":
                   draw_pie(A, x + 3, y - 4, coe*Energy2019.loc[c, MkrSize], colorss = lcolor, ax = ax)
            elif c == "Syria":
                   draw_pie(A, x - 1.3, y, coe*Energy2019.loc[c, MkrSize], colorss = lcolor, ax = ax)
            elif c == "Netherlands":
                   draw_pie(A, x, y + 2, coe*Energy2019.loc[c, MkrSize], colorss = lcolor, ax = ax)
            elif c == "Belgium":
                   draw_pie(A, x - 1, y + 1, coe*Energy2019.loc[c, MkrSize], colorss = lcolor, ax = ax)
            elif c == "Croatia":                   
                    draw_pie(A, x, y + 0.8, coe*Energy2019.loc[c, MkrSize], colorss = lcolor, ax = ax)
            else:
                    draw_pie(A, x, y, coe*Energy2019.loc[c, MkrSize], colorss = lcolor, ax = ax)

## Good Idea to visualize gradient bar color. Thanks Andrew!
## https://stackoverflow.com/questions/61460814/color-cartopy-map-countries-according-to-given-values

dummy_scat = ax.scatter(Energy2019.loc[p,"energy_per_gdp"],
                        Energy2019.loc[p,"energy_per_gdp"],
                        c=Energy2019.loc[p,"energy_per_gdp"], cmap=cmap, zorder=0)

## bax is an axes to display color gradient
plt.colorbar(mappable = dummy_scat, cax = bax)

## Title for colorbar
bax.set_ylabel("Total Consumption [GWh] per GDP [Millions USD]")

## Remove tick marks for the sake of aesthetics
ax_tech.set_xticks([])
ax_tech.set_yticks([])
ax_cir.set_xticks([])
ax_cir.set_yticks([])

# Show infography
plt.show()






