================================================================================
  visualize_firms_dataset.py – User Guide & Configuration Reference
================================================================================

This document explains how to use the NASA FIRMS fire data visualization and
analysis script, including authentication, changing regions/states, filters,
and output options.


--------------------------------------------------------------------------------
0. INTERACTIVE MODE (no flags needed)
--------------------------------------------------------------------------------

  Run the script with no arguments to get prompted for options:
    python visualize_firms_dataset.py

  Or explicitly:
    python visualize_firms_dataset.py -i
    python visualize_firms_dataset.py --interactive

  You will be prompted for:
    - Date range (or last N days)
    - Area(s): comma-separated (e.g. 1,2,3 = California, Eaton, Palisades)
    - Data source (Auto / Archive / NRT)
    - Plot type(s): comma-separated (e.g. 1,2,3 = all visualizations)
    - Filters, outputs, etc.

  Press Enter to accept the default for each prompt.


--------------------------------------------------------------------------------
1. PURPOSE & FEATURES
--------------------------------------------------------------------------------

The script fetches NASA FIRMS (Fire Information for Resource Management System)
fire detection data, subsets it by region (bounding box, shapefile, or WKT
polygon), applies optional filters, adds derived features for ML, and produces
CSV exports plus optional maps (PNG, Shapefile, KML).

Features:
  - FIRMS API integration with fallback to sample data if no API key
  - Filters: confidence (n/h/l), minimum FRP, day/night (D/N)
  - Subsetting: bbox, WKT polygon, or shapefile (e.g. by US state name)
  - Derived columns: acq_datetime, footprint corners, footprint area
  - Optional Earth Engine enrichment: elevation, NDVI, population density, land cover
  - Outputs: CSV (always), optional Shapefile, KML, and PNG fire maps


--------------------------------------------------------------------------------
2. INSTALLATION & DEPENDENCIES
--------------------------------------------------------------------------------

Required (core):
  pip install matplotlib pandas numpy requests

Optional:
  pip install geopandas shapely     # For --shapefile and --wkt-polygon subsetting
  pip install contextily            # For --basemap (background map tiles)
  pip install earthengine-api       # For --enrich-earth-engine

From the script directory, you can install from requirements.txt if present:
  pip install -r requirements.txt


--------------------------------------------------------------------------------
3. AUTHENTICATION
--------------------------------------------------------------------------------

3.1 FIRMS MAP_KEY (required for real API data)

  Without a MAP_KEY, the script uses fallback sample data only. To get full
  data for your region and date range:

  a) Obtain a free key:
     https://firms.modaps.eosdis.nasa.gov/api/map_key

  b) Provide the key in one of two ways:
     - Environment variable:
         Windows CMD:     set FIRMS_MAP_KEY=your_key_here
         Windows PowerShell: $env:FIRMS_MAP_KEY="your_key_here"
         Linux/Mac:       export FIRMS_MAP_KEY=your_key_here
     - Interactive prompt: When you run the script without the env var set,
       it will ask you to paste the MAP_KEY (or Enter to use sample only)

  c) Skip key verification if desired:
     python visualize_firms_dataset.py --no-verify-key


3.2 Earth Engine (only for --enrich-earth-engine)

  If you use --enrich-earth-engine, you must authenticate with Google Earth
  Engine first:

  a) Install: pip install earthengine-api
  b) Register at https://code.earthengine.google.com and create a project
  c) Run once: earthengine authenticate
  d) Set your Google Cloud project (required by newer earthengine-api):
       set EE_PROJECT=your-gcp-project-id     (Windows CMD)
       $env:EE_PROJECT="your-gcp-project-id"   (PowerShell)
     Or: earthengine set_project your-gcp-project-id
  e) Credentials are cached (typically in ~/.config/earthengine/)


--------------------------------------------------------------------------------
4. CHANGING REGIONS: STATES, BBOX, SHAPEFILE, WKT
--------------------------------------------------------------------------------

4.1 Default behavior

  By default, the script uses California’s bounding box:
    west, south, east, north = -124.5, 32.5, -114.1, 42.0

  Use --area eaton or --area palisades for focused LA-area subsets.


4.2 Custom bounding box (--bbox)

  Override the region with any bounding box (west,south,east,north):

  python visualize_firms_dataset.py --bbox -85,-57,-32,14
  python visualize_firms_dataset.py --bbox -124.5,32.5,-114.1,42.0

  Common US regions (approximate):
    California:     -124.5,32.5,-114.1,42.0
    Pacific Palisades (2025 fire extent): -118.90,33.96,-118.38,34.20
    Texas:          -106.6,25.8,-93.5,36.5
    Oregon:         -124.6,41.9,-116.5,46.3
    Washington:     -124.9,45.5,-116.9,49.0
    Arizona:        -114.8,31.3,-109.0,37.0
    Colorado:       -109.1,36.9,-102.0,41.0
    Montana:        -116.1,44.3,-104.0,49.0


4.3 Shapefile by state (--shapefile --state)

  Use a US states (or similar) shapefile and filter by state name:

  python visualize_firms_dataset.py --shapefile path/to/cb_2018_us_state_20m.shp --state "California"
  python visualize_firms_dataset.py --shapefile states.shp --state "Oregon"

  Notes:
  - The script expects a column named "NAME" for the state name. If your
    shapefile uses a different column (e.g. "STATE_NAME"), you need to edit
    the script and change the name_col default in subset_by_shapefile()
    (around line 144).
  - State name must match exactly (case-sensitive).
  - Shapefile path can be relative or absolute.


4.4 WKT polygon (--wkt-polygon)

  Define your region as a WKT polygon:

  python visualize_firms_dataset.py --wkt-polygon "POLYGON((-123 38,-121 38,-121 40,-123 40,-123 38))"

  Format: POLYGON((lon1 lat1, lon2 lat2, ...)) – first point = last point for
  a closed polygon.


--------------------------------------------------------------------------------
5. DATE & TIME RANGE
--------------------------------------------------------------------------------

  --start-date and --end-date:
    python visualize_firms_dataset.py --start-date 2025-01-07 --end-date 2025-01-31
    python visualize_firms_dataset.py --start-date 2025-01-08 --end-date 2025-01-08  # single day

  Older dates: Use --use-archive or rely on default (tries archive if NRT empty).

  Note: If you get 2023 data when requesting 2025, the API returned empty and the
  script was falling back to sample data. This is now fixed—you'll see
  "No FIRMS data for requested date range" instead. Ensure you have a valid
  FIRMS MAP_KEY and try --use-archive for historical dates.

  --days (used when no date range; default 5, max 10):
    python visualize_firms_dataset.py --days 3
    python visualize_firms_dataset.py --days 10


--------------------------------------------------------------------------------
6. DATA FILTERS (CONFIDENCE, FRP, DAY/NIGHT)
--------------------------------------------------------------------------------

  --confidence: Comma-separated list (n=normal, h=high, l=low)
    python visualize_firms_dataset.py --confidence n,h

  --min-frp: Minimum Fire Radiative Power
    python visualize_firms_dataset.py --min-frp 5

  --daynight: D = day only, N = night only
    python visualize_firms_dataset.py --daynight D

  Example combining filters:
    python visualize_firms_dataset.py --confidence n,h --min-frp 5 --daynight D


--------------------------------------------------------------------------------
7. OUTPUT OPTIONS
--------------------------------------------------------------------------------

  --output-dir PATH          Change output folder (default: firms_output/)
  --save-shapefile           Also save output as .shp
  --save-kml                 Also save output as .kml
  --save-hdf5                Also save output as HDF5 (.h5)
  --time-plot                Use time-based colored map instead of FRP
  --fire-mask                Plot fire footprint polygons (detected area mask) instead of points
  --area AREA (repeatable)  Area: eaton, california, palisades (e.g. --area eaton --area palisades)
  --points                  Plot points (FRP-colored)
  --basemap                  Add basemap tiles to time plot / fire mask (needs contextily)
  --enrich-earth-engine      Add elevation, NDVI, population, land cover (needs EE auth)


--------------------------------------------------------------------------------
8. FIRMS SOURCE (NRT vs ARCHIVE)
--------------------------------------------------------------------------------

  By default, the script tries NRT (Near Real-Time) first, then archive (SP)
  sources. This supports both recent (~10 days) and older historical data.

  NRT sources (recent data):  VIIRS_SNPP_NRT, VIIRS_NOAA20_NRT, VIIRS_NOAA21_NRT, MODIS_NRT
  Archive sources (historical): VIIRS_SNPP_SP, VIIRS_NOAA20_SP, MODIS_SP
    (Archive extends: MODIS to Nov 2000, VIIRS S-NPP to Jan 2012, etc.)

  --source: Force a specific source
    python visualize_firms_dataset.py --source VIIRS_SNPP_NRT
    python visualize_firms_dataset.py --source VIIRS_SNPP_SP  # archive for old dates

  --use-archive: Use archive/SP sources only (skip NRT; for known historical dates)
    python visualize_firms_dataset.py --start-date 2025-01-08 --end-date 2025-01-08 --use-archive

  --nrt-only: Use NRT sources only (skip archive; for recent data only)


--------------------------------------------------------------------------------
9. EXAMPLE COMMANDS
--------------------------------------------------------------------------------

  Basic run (California, last 5 days, default filters):
    python visualize_firms_dataset.py

  Oregon, custom date range:
    python visualize_firms_dataset.py --start-date 2025-01-07 --end-date 2025-01-31 --bbox -124.6,41.9,-116.5,46.3

  California via shapefile:
    python visualize_firms_dataset.py --shapefile cb_2018_us_state_20m.shp --state "California"

  Custom polygon, strict filters, all outputs:
    python visualize_firms_dataset.py --wkt-polygon "POLYGON((-123 38,-121 38,-121 40,-123 40,-123 38))" --confidence n,h --min-frp 5 --daynight D --save-shapefile --save-kml --time-plot --basemap --enrich-earth-engine

  Multiple areas and visualizations:
    python visualize_firms_dataset.py --area eaton --area palisades --fire-mask --points
    python visualize_firms_dataset.py --area california --fire-mask --time-plot --points

  Use sample data only (no MAP_KEY):
    python visualize_firms_dataset.py
    (when prompted for MAP_KEY, press Enter)


--------------------------------------------------------------------------------
10. OUTPUT FILES
--------------------------------------------------------------------------------

  Default outputs in firms_output/:
    firms_california.csv     (or firms_polygon.csv / firms_shapefile_StateName.csv)
    firms_california.png     (fire map)

  Eaton (--area eaton): firms_eaton.csv, firms_eaton.png
  Palisades (--area palisades): firms_palisades.csv, firms_palisades_mask.png

  With --save-shapefile:  .shp, .shx, .dbf, .cpg
  With --save-kml:        .kml
  With --save-hdf5:       .h5 (requires: pip install tables)
  With --time-plot:       {base}_time.png
  With --fire-mask:       {base}_mask.png (footprint polygons = detected fire area;
                       note: FIRMS = active hotspots, not post-fire burned perimeters)


--------------------------------------------------------------------------------
11. EDITING THE SCRIPT DIRECTLY
--------------------------------------------------------------------------------

  Bounding boxes:
    CALIFORNIA_BBOX, EATON_BBOX, and PALISADES_BBOX are defined near the top.
    Edit these to change the default California, Eaton, and Pacific Palisades regions.

  Shapefile state column:
    In subset_by_shapefile(), the default name_col is "NAME". If your shapefile
    uses a different column, change the default or pass it via the function
    (requires code change to expose --name-col in argparse).

  Earth Engine batch size:
    EE_BATCH_SIZE controls how many points are sampled per EE request.
    Default is around 500; reduce if you get timeouts.


--------------------------------------------------------------------------------
12. TROUBLESHOOTING
--------------------------------------------------------------------------------

  "No FIRMS data retrieved"
    - Check MAP_KEY; without it only sample data is used.
    - Verify date range and bbox are valid.
    - Ensure FIRMS has data for your region/dates (fire activity).

  "State 'X' not found in shapefile"
    - Confirm the exact state name and column (usually "NAME").

  "WARNING: Polygon subsetting skipped"
    - Install: pip install geopandas shapely

  "WARNING: --enrich-earth-engine ignored"
    - Install: pip install earthengine-api
    - Run: earthengine authenticate

  "HDF5 save skipped. Install: pip install tables"
    - Required for --save-hdf5: pip install tables

  "ee.Initialize: no project found"
    - Set EE_PROJECT env var to your Google Cloud project ID
    - Or run: earthengine set_project your-gcp-project-id
    - Register a project at https://code.earthengine.google.com

  EE "sampling error" or timeouts
    - Try reducing EE_BATCH_SIZE in the script.
    - Check your Earth Engine quota and network.


--------------------------------------------------------------------------------
13. QUICK REFERENCE
--------------------------------------------------------------------------------

  Key URLs:
    FIRMS MAP_KEY:  https://firms.modaps.eosdis.nasa.gov/api/map_key
    FIRMS API docs: https://firms.modaps.eosdis.nasa.gov/api/area/
    FIRMS data ingest: https://firms.modaps.eosdis.nasa.gov/content/academy/data_ingest/

  Bbox format: west,south,east,north
  Day range: 1–10 when using --days

================================================================================
End of User Guide
================================================================================
