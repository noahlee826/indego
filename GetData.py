import json, urllib.request, configparser, utils
from census import Census
from us import states


def get_station_info():
    data = urllib.request.urlopen("https://gbfs.bcycle.com/bcycle_indego/station_information.json").read()
    station_info_json = json.loads(data)
    # print(json.dumps(station_info_json, indent=4, sort_keys=False))
    station_list = station_info_json['data']['Stations']
    # print(station_list)

    # Station IDs are given in the format 'bcycle_indego_NNNN' where N is [0..9]
    # We only want the numeric part
    # station_latlons = {stn['station_id'].rsplit('_', maxsplit=1)[1]: (stn['lat'], stn['lon']) for stn in station_list}
    # print(station_latlons)
    for station in station_list:
        station['station_id'] = station['station_id'].rsplit('_', maxsplit=1)[1]
    return station_list


def get_FCC_block_info(lat, lon):
    # lat, lon = latlon
    base_api_url = "https://geo.fcc.gov/api/census/block/find?"
    api_request = base_api_url \
                    + 'latitude=' + str(lat) + '&' \
                    + 'longitude=' + str(lon) + '&' \
                    + '&format=json'
    data = urllib.request.urlopen(api_request).read()
    return json.loads(data)


def get_station_characteristics(census=None):
    if census is None:
        census = Census(utils.get_config_val('API Keys', 'census_api'), year=2017)

    blockgroup_densities = get_census_pdb_pop_density()
    station_list = get_station_info()

    for station in station_list:
        # print(station)
        block_info = get_FCC_block_info(station['lat'], station['lon'])
        block = block_info['Block']['FIPS'][12:]
        blockgroup = block_info['Block']['FIPS'][11]  # The 11th digit identifies the block group
        tract = block_info['Block']['FIPS'][5:11]
        county = block_info['County']['FIPS'][2:5]
        state = block_info['State']['FIPS']

        blockgroup_full = block_info['Block']['FIPS'][:12]  # Need the first 11 digits to identify state, county, blockgroup

        station['block_FIPS'] = block
        station['blockgroup_FIPS'] = blockgroup
        station['tract_FIPS'] = tract
        station['county_FIPS'] = county
        station['state_FIPS'] = state
        station['pop_density'] = blockgroup_densities[blockgroup_full]
        station['pct_in_poverty'] = get_acs5_under_poverty(census, state, county, tract, blockgroup)
        print(station)
    print(station_list)


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
    blockgroups_densities = {bg['state'] + bg['county'] + bg['tract'] + bg['block group'] :
                             float(bg['Tot_Population_ACS_12_16']) / float(bg['LAND_AREA'])
                             for bg in blockgroups_list}
    return blockgroups_densities


def get_acs5_under_poverty(census, state, county, tract, blockgroup):
    #TODO maybe hammer this out to only make one API call instead of one call per blockgroup

    # For complete list of available variables, see
    # (WARNING: LARGE PAGE) https://api.census.gov/data/2017/acs/acs5/variables.html

    fields = ('B00001_001E',  # Estimate!!Total
              'B23024_002E')  # Estimate!!Total!!Income in the past 12 months below poverty level
    data = census.acs5.state_county_blockgroup(fields, state, county, blockgroup, tract=tract) # returns list with 1 elt, a dict containing a key for each value provided.
    # print(data)
    tot_pop = data[0][fields[0]]
    num_poverty = data[0][fields[1]]
    pct_in_poverty = (num_poverty / tot_pop) if tot_pop is not None and tot_pop > 0 else 0
    return pct_in_poverty


# foo = get_census_pdb_pop_density()
# print(foo)
# print(get_acs5_under_poverty(c, '42', '101', '000402', '2'))
c = Census(utils.get_config_val('API Keys', 'census_api'), year=2017)
get_station_characteristics(c)
