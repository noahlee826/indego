import json, urllib.request, configparser, utils
from census import Census
from us import states
from collections import defaultdict

# returns list of stations
def get_indego_station_info(include_station_ids=None):
    print('get indego info using station list:', include_station_ids)
    include_all_stations = False
    if include_station_ids is None:
        include_all_stations = True

    data = urllib.request.urlopen("https://gbfs.bcycle.com/bcycle_indego/station_information.json").read()
    station_info_json = json.loads(data)
    # print(json.dumps(station_info_json, indent=4, sort_keys=False))
    station_list = station_info_json['data']['stations']
    # print(station_list)

    # Station IDs are given in the format 'bcycle_indego_NNNN' where N is [0..9]
    # We only want the numeric part
    # station_latlons = {stn['station_id'].rsplit('_', maxsplit=1)[1]: (stn['lat'], stn['lon']) for stn in station_list}
    # print(station_latlons)

    for station in station_list:
        station['station_id'] = station['station_id'].rsplit('_', maxsplit=1)[1]
    # result_list = [station for station in station_list if station['station_id'] in include_station_ids
    #                                                    or include_all_stations]
    result_list = []
    for station in station_list:
        if include_all_stations or station['station_id'] in include_station_ids:
            result_list.append(station)
    return result_list


def get_FCC_block_info(lat, lon):
    # lat, lon = latlon
    base_api_url = "https://geo.fcc.gov/api/census/block/find?"
    api_request = base_api_url \
                  + 'latitude=' + str(lat) + '&' \
                  + 'longitude=' + str(lon) + '&' \
                  + '&format=json'
    data = urllib.request.urlopen(api_request).read()
    return json.loads(data)


def get_station_characteristics(census=None, include_station_ids=None):
    print('get station chars using station list:', include_station_ids)
    if census is None:
        census = Census(utils.get_config_val('API Keys', 'census_api'), year=2017)

    blockgroup_densities, blockgroup_poverties, blockgroup_incomes = get_census_pdb_pop_density_poverty_income()
    station_list = get_indego_station_info(include_station_ids)

    for station in station_list:
        # print(station)
        block_info = get_FCC_block_info(station['lat'], station['lon'])
        block = block_info['Block']['FIPS'][12:]
        blockgroup = block_info['Block']['FIPS'][11]  # The 11th digit identifies the block group
        tract = block_info['Block']['FIPS'][5:11]
        county = block_info['County']['FIPS'][2:5]
        state = block_info['State']['FIPS']

        blockgroup_full = block_info['Block']['FIPS'][:12]  # Need the first 11 digits to identify state/county/blockgrp

        station['block_FIPS'] = block
        station['blockgroup_FIPS'] = blockgroup
        station['tract_FIPS'] = tract
        station['county_FIPS'] = county
        station['state_FIPS'] = state
        station['pop_density'] = blockgroup_densities[blockgroup_full]
        # station['pct_in_poverty'], station['med_income'] = get_acs5_data(census, state, county, tract, blockgroup)
        station['pct_in_poverty'] = blockgroup_poverties[blockgroup_full]
        station['med_income'] = blockgroup_incomes[blockgroup_full]
        # print(station)
    print(station_list)
    return station_list


# Returns a dict of the form {blockgroup: population density
def get_census_pdb_pop_density():
    # For help building this query:
    # https://www.census.gov/data/developers/guidance/api-user-guide.Query_Examples.html
    # Uses 2016 ACS estimate for population and area
    api_url = 'https://api.census.gov/data/2018/pdb/blockgroup?get=County_name,State_name,Tot_Population_ACS_12_16,LAND_AREA&for=block%20group:*&in=state:42%20county:101'
    data = urllib.request.urlopen(api_url).read()
    json_data = json.loads(data)
    labels = json_data[0]

    blockgroups_list = [dict(zip(labels, row)) for row in json_data[1:]]
    blockgroups_densities = {bg['state'] + bg['county'] + bg['tract'] + bg['block group']:
                                 float(bg['Tot_Population_ACS_12_16']) / float(bg['LAND_AREA'])
                             for bg in blockgroups_list}
    return blockgroups_densities


# Returns a 3-tuple of the form ({blockgroup: population density),
#                                {blockgroup: percent in poverty},
#                                {blockgroup: median income})
def get_census_pdb_pop_density_poverty_income():
    # For help building this query:
    # https://www.census.gov/data/developers/guidance/api-user-guide.Query_Examples.html
    # Uses 2016 ACS estimate for population and area
    api_url = 'https://api.census.gov/data/2018/pdb/blockgroup?get=County_name,State_name,Tot_Population_ACS_12_16,LAND_AREA,Prs_Blw_Pov_Lev_ACS_12_16,Pov_Univ_ACS_12_16,pct_Prs_Blw_Pov_Lev_ACS_12_16,Med_HHD_Inc_BG_ACS_12_16&for=block%20group:*&in=state:42%20county:101'
    data = urllib.request.urlopen(api_url).read()
    json_data = json.loads(data)
    labels = json_data[0]

    blockgroups_list = [dict(zip(labels, row)) for row in json_data[1:]]
    blockgroups_densities = {bg['state'] + bg['county'] + bg['tract'] + bg['block group']:
                             int(bg['Tot_Population_ACS_12_16']) / float(bg['LAND_AREA'])
                             for bg in blockgroups_list}
    blockgroups_poverties = {bg['state'] + bg['county'] + bg['tract'] + bg['block group']:
                             int(bg['Prs_Blw_Pov_Lev_ACS_12_16']) / int(bg['Pov_Univ_ACS_12_16'])
                                if int(bg['Pov_Univ_ACS_12_16']) != 0
                                else 0
                             for bg in blockgroups_list}
    blockgroups_incomes = {bg['state'] + bg['county'] + bg['tract'] + bg['block group']:
                           int(bg['Med_HHD_Inc_BG_ACS_12_16'].strip('$').replace(',', ''))  # Get rid of $ and , to convert to float
                                if bg['Med_HHD_Inc_BG_ACS_12_16']  # Income string is empty
                                else 0
                           for bg in blockgroups_list}

    return blockgroups_densities, blockgroups_poverties, blockgroups_incomes


# returns tuple of (percent in poverty, median income)
def get_acs5_data(census, state, county, tract, blockgroup):
    # TODO maybe hammer this out to only make one API call instead of one call per blockgroup

    # For complete list of available variables, see
    # (WARNING: LARGE PAGE) https://api.census.gov/data/2017/acs/acs5/variables.html

    total_pop_field = 'B01003_001E'  # Estimate!!Total	TOTAL POPULATION
    num_poverty_field = 'B23024_002E'  # Estimate!!Total!!Income in the past 12 months below poverty level

    # Careful! Next line returns a number of people (households?), who have income. Other _002 through _017 break it into ranges
    med_income_field = 'B19001_001E'  # Estimate!!Total	    HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN 2017 INFLATION-ADJUSTED DOLLARS)

    # B21004_001E # another possible median income
    bg_fields = (total_pop_field,
                 num_poverty_field)
    tract_fields = (med_income_field)

    # Returns list with 1 elt, a dict containing a key for each value provided.
    bg_data = census.acs5.state_county_blockgroup(bg_fields, state, county, blockgroup, tract=tract)
    tract_data = census.acs5.state_county_tract(tract_fields, state, county, tract)
    # print(tract_data)
    # print(data)
    tot_pop = bg_data[0][total_pop_field]
    num_poverty = bg_data[0][num_poverty_field]
    med_income = tract_data[0][med_income_field]
    pct_in_poverty = (num_poverty / tot_pop) if tot_pop is not None and tot_pop > 0 else 0
    return pct_in_poverty, med_income


# fields is a list of column names in the table described at the API documentation below
# tracts is list of 3-digit census tracts represented as strings
# API Documentation:
# https://cityofphiladelphia.github.io/carto-api-explorer/#opa_properties_public
# http://metadata.phila.gov/#home/datasetdetails/5543865f20583086178c4ee5/
def get_opa_data(fields=None, tracts=None):
    base_url = 'https://phl.carto.com/api/v2/sql?q=SELECT'
    cols = '*' if fields is None \
        else ', '.join(fields)
    FROM = 'FROM'
    table = 'opa_properties_public'
    WHERE = 'WHERE'
    census_tract = 'census_tract'
    IN = 'IN'
    tract_value_list = '()' if tracts is None \
        else str(tuple(tracts))
    tract_clause = '' if tracts is None else ' WHERE census_tract IN {0}'.format(tract_value_list)

    query = "https://phl.carto.com/api/v2/sql?q=SELECT {0} FROM opa_properties_public{1}".format(cols, tract_clause)

    print(query)
    # data = urllib.request.urlopen(query).read()
    # opa_data = json.loads(data)


def get_station_FIPS(census=None, include_station_ids=None):
    if census is None:
        census = Census(utils.get_config_val('API Keys', 'census_api'), year=2017)

    station_list = get_indego_station_info(include_station_ids)

    for station in station_list:
        # print(station)
        block_info = get_FCC_block_info(station['lat'], station['lon'])
        block = block_info['Block']['FIPS'][12:]
        blockgroup = block_info['Block']['FIPS'][11]  # The 11th digit identifies the block group
        tract = block_info['Block']['FIPS'][5:11]
        county = block_info['County']['FIPS'][2:5]
        state = block_info['State']['FIPS']

        blockgroup_full = block_info['Block']['FIPS'][
                          :12]  # Need the first 11 digits to identify state, county, blockgroup

        station['block_FIPS'] = block
        station['blockgroup_FIPS'] = blockgroup
        station['tract_FIPS'] = tract
        station['county_FIPS'] = county
        station['state_FIPS'] = state
    return station_list


def get_zoning_data(include_station_ids=None):
    station_list = get_indego_station_info(include_station_ids)
    station_list_dds = [defaultdict(int, station) for station in station_list]
    global_zoning_groups = set()

    for station in station_list_dds:
        lat = station['lat']
        lon = station['lon']
        radius = 75.0  # Most Philadelphia blocks are about 150m long, so search uses within half of a block
        # radius = 400  # Use roughly 1/4 mile as that is how far most American will walk to an amenity
                        # and the distance Indego uses to asses nearby population and jobs
                        # qv page 6: http://www.phillyotis.com/wp-content/uploads/2018/10/2018_IndegoPlan_Full_Final.pdf
        nearby_zoning_groups = get_nearby_zoning_groups(lat, lon, radius, use_zoning_code=False)
        for zg in nearby_zoning_groups:
            station[zg] = 1
            global_zoning_groups.add(zg)

    for station in station_list_dds:
        print(station)
    return station_list_dds, sorted(global_zoning_groups)


# Returns a string that is the zoning group of closest zoned plot
# Caveat: breaks up "Special Purpose" zoning group into its specific zones
# See zoning quick reference guide:
# https://www.phila.gov/CityPlanning/resources/Publications/Philadelphia%20Zoning%20Code_Quick%20Reference%20Manual.pdf
def get_nearest_zoning_group(lat, lon):
    # Returns properties whose footprints are entirely within 500m of pt
    # query = str('https://services.arcgis.com/fLeGjb7u4uXqeF9q/arcgis/rest/services/Zoning_BaseDistricts/FeatureServer/0/query?where=1%3D1&objectIds=&time=&geometry=%7B"x"+%3A+{0}%2C+"y"+%3A+{1}%2C+"spatialReference"%3A%7B"wkid"+%3A+4326%7D%7D&geometryType=esriGeometryPoint&inSR=&spatialRel=esriSpatialRelContains&resultType=none&distance=500.0&units=esriSRUnit_Meter&returnGeodetic=false&outFields=*&returnGeometry=true&returnCentroid=false&multipatchOption=xyFootprint&maxAllowableOffset=&geometryPrecision=&f=pjson&token=')\
    # for obj in zgroup_data['features']:
    #     zgroup = obj['attributes']['ZONINGGROUP'] # Name of zoning group (Commercial, Residential, etc.)
    #     zgroup_areas[zgroup] += obj['attributes']['Shape__Area']

    radius = 1.0  # Many stations are located in the public right-of-way, just barely outside of a zoned plot
                  # So we need to include a small radius around the point to intersect a zoned plot.
                  # Units are meters.
    query = str('https://services.arcgis.com/fLeGjb7u4uXqeF9q/arcgis/rest/services/Zoning_BaseDistricts/FeatureServer/0/query?where=1%3D1&objectIds=&time=&geometry=%7B"x"+%3A+{0}%2C+"y"+%3A+{1}%2C+"spatialReference"%3A%7B"wkid"+%3A+4326%7D%7D&geometryType=esriGeometryPoint&inSR=&spatialRel=esriSpatialRelIntersects&resultType=none&distance={2}&units=esriSRUnit_Meter&returnGeodetic=false&outFields=*&returnGeometry=true&returnCentroid=false&multipatchOption=xyFootprint&maxAllowableOffset=&geometryPrecision=&outSR=&datumTransformation=&applyVCSProjection=false&returnIdsOnly=false&returnUniqueIdsOnly=false&returnCountOnly=false&returnExtentOnly=false&returnQueryGeometry=false&returnDistinctValues=false&orderByFields=&groupByFieldsForStatistics=&outStatistics=&having=&resultOffset=&resultRecordCount=&returnZ=false&returnM=false&returnExceededLimitFeatures=true&quantizationParameters=&sqlFormat=none&f=pjson&token=') \
        .format(lon, lat, str(radius))  # The API is in (x, y) format, which is {lon, lat)
    print(query)
    data = urllib.request.urlopen(query).read()
    json_data = json.loads(data)

    # May need to update query to enlarge the radius to account for stations further from a zoned plot.
    while len(json_data['features']) == 0:  # Each itme in json_data['features'] is a zoned plot.
        radius *= 1.5
        query = str(
            'https://services.arcgis.com/fLeGjb7u4uXqeF9q/arcgis/rest/services/Zoning_BaseDistricts/FeatureServer/0/query?where=1%3D1&objectIds=&time=&geometry=%7B"x"+%3A+{0}%2C+"y"+%3A+{1}%2C+"spatialReference"%3A%7B"wkid"+%3A+4326%7D%7D&geometryType=esriGeometryPoint&inSR=&spatialRel=esriSpatialRelIntersects&resultType=none&distance={2}&units=esriSRUnit_Meter&returnGeodetic=false&outFields=*&returnGeometry=true&returnCentroid=false&multipatchOption=xyFootprint&maxAllowableOffset=&geometryPrecision=&outSR=&datumTransformation=&applyVCSProjection=false&returnIdsOnly=false&returnUniqueIdsOnly=false&returnCountOnly=false&returnExtentOnly=false&returnQueryGeometry=false&returnDistinctValues=false&orderByFields=&groupByFieldsForStatistics=&outStatistics=&having=&resultOffset=&resultRecordCount=&returnZ=false&returnM=false&returnExceededLimitFeatures=true&quantizationParameters=&sqlFormat=none&f=pjson&token=') \
            .format(lon, lat, str(radius))  # The API is in (x, y) format, which is {lon, lat)
        print('Incremented Distance... dist = {0}'.format(str(radius)))
        print(query)
        data = urllib.request.urlopen(query).read()
        json_data = json.loads(data)

    # Use the first feature, typically the closest if there's more than one
    zoning_group = json_data['features'][0]['attributes']['ZONINGGROUP']

    # Further specify SP zoning group
    if zoning_group == 'Special Purpose':
        zoning_group = json_data['features'][0]['attributes']['CODE']
    return zoning_group


# radius units are meters
# if use_zoning_Code == True, analysis will use the more-specific zoning code instead of general use categories
# Returns set of zoning groups that appear within radius meters of (lat, lon)
# Caveat: breaks up "Special Purpose" zoning group into its specific zones
# See zoning quick reference guide:
# https://www.phila.gov/CityPlanning/resources/Publications/Philadelphia%20Zoning%20Code_Quick%20Reference%20Manual.pdf
def get_nearby_zoning_groups(lat, lon, radius=0.0, use_zoning_code=False, split_special_purpose=True):
    zoning_type = 'CODE' if use_zoning_code else 'ZONINGGROUP'
    query = str(
        'https://services.arcgis.com/fLeGjb7u4uXqeF9q/arcgis/rest/services/Zoning_BaseDistricts/FeatureServer/0/query?where=1%3D1&objectIds=&time=&geometry=%7B"x"+%3A+{0}%2C+"y"+%3A+{1}%2C+"spatialReference"%3A%7B"wkid"+%3A+4326%7D%7D&geometryType=esriGeometryPoint&inSR=&spatialRel=esriSpatialRelIntersects&resultType=none&distance={2}&units=esriSRUnit_Meter&returnGeodetic=false&outFields=*&returnGeometry=true&returnCentroid=false&multipatchOption=xyFootprint&maxAllowableOffset=&geometryPrecision=&outSR=&datumTransformation=&applyVCSProjection=false&returnIdsOnly=false&returnUniqueIdsOnly=false&returnCountOnly=false&returnExtentOnly=false&returnQueryGeometry=false&returnDistinctValues=false&orderByFields=&groupByFieldsForStatistics=&outStatistics=&having=&resultOffset=&resultRecordCount=&returnZ=false&returnM=false&returnExceededLimitFeatures=true&quantizationParameters=&sqlFormat=none&f=pjson&token=') \
        .format(lon, lat, str(radius))  # The API takes coords in (x, y) format, which is (lon, lat)
    print(query)
    data = urllib.request.urlopen(query).read()
    json_data = json.loads(data)
    zoning_groups = set()
    for plot in json_data['features']:
        zoning = plot['attributes'][zoning_type]
        if zoning == 'Special Purpose':  # Use specific zoning code (SPINS, SPENT, etc.) instead of 'Special Purpose'
            zoning = plot['attributes']['CODE']
        zoning_groups.add(zoning)
    # zoning_groups = set([plot['attributes'][attrib] for plot in json_data['features']])
    return zoning_groups



# stations = get_station_FIPS()
# tracts = set([str(station['tract_FIPS'])[-3:] for station in stations])
# important_cols = ['category_code',
#                   'category_code_description',
#                   'census_tract',
#                   'frontage',
#                   'total_area',
#                   'zoning'
#                   ]
# get_opa_data(fields=important_cols, tracts=tracts)


# foo = get_census_pdb_pop_density()
# print(foo)
# print(get_acs5_under_poverty(c, '42', '101', '000402', '2'))
# c = Census(utils.get_config_val('API Keys', 'census_api'), year=2017)
# get_station_characteristics(c)
