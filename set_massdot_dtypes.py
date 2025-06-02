import json
import pandas as pd
import geopandas as gpd
from pathlib import Path

def set_massdot_roads_dtypes():
    """
    Sets the correct data types for MassDOT Roads GeoJSON dataset
    based on the official documentation and field definitions.
    """
    
    # Define data types based on MassDOT documentation
    # Fields added by MassGIS
    massgis_fields = {
        # Numeric fields
        'OBJECTID': 'int64',
        'CLASS': 'int64',  # Values 1-7 for road classification
        'ADMIN_TYPE': 'int64',  # 0=Not applicable, 1=Interstate, 2=US Highway, 3=State Highway
        'RDTYPE': 'int64',  # Road type for display purposes
        'LENGTH_MI': 'float64',  # Length in miles
        'LENGTH_FT': 'float64',  # Length in feet
        
        # String fields
        'MGIS_TOWN': 'str',  # Town name
        'STREET_NAME': 'str',  # Street name
        'RT_NUMBER': 'str',  # Route number
        'ALTRTNUM1': 'str',  # Alternate route number 1
        'ALTRTNUM2': 'str',  # Alternate route number 2
        'ALTRTNUM3': 'str',  # Alternate route number 3
        'ALTRTNUM4': 'str',  # Alternate route number 4
        'ALTRT1TYPE': 'str',  # Alternate route 1 type
    }
    
    # MassDOT Road Inventory fields
    massdot_fields = {
        # Identification fields
        'ROADINVENTORY_ID': 'str',
        'CRN': 'str',  # County Route Number
        'ROADSEGMENT_ID': 'str',
        
        # Linear referencing fields
        'FROMMEASURE': 'float64',
        'TOMEASURE': 'float64',
        'ASSIGNEDLENGTH': 'float64',
        'ASSIGNED_LENGTH_SOURCE': 'str',
        
        # Street information
        'STREETLIST_ID': 'str',
        'STREETNAME': 'str',
        'STREETNAME_ID': 'str',
        'CITY': 'str',
        'COUNTY': 'str',
        'MUNICIPALSTATUS': 'str',
        
        # From/To information
        'FROMENDTYPE': 'str',
        'FROMSTREETNAME': 'str',
        'FROMCITY': 'str',
        'FROMSTATE': 'str',
        'TOENDTYPE': 'str',
        'TOSTREETNAME': 'str',
        'TOCITY': 'str',
        'TOSTATE': 'str',
        
        # Route information
        'MILEAGECOUNTED': 'str',
        'ROUTEKEY': 'str',
        'ROUTEFROM': 'float64',
        'ROUTETO': 'float64',
        'ROUTESYSTEM': 'str',
        'ROUTENUMBER': 'str',
        'SUBROUTE': 'str',
        'ROUTEDIRECTION': 'str',
        'ROUTETYPE': 'str',
        'ROUTEQUALIFIER': 'str',
        
        # Administrative fields
        'RPA': 'str',  # Regional Planning Agency
        'MPO': 'str',  # Metropolitan Planning Organization
        'MASSDOTHIGHWAYDISTRICT': 'int64',
        
        # Classification fields
        'URBANTYPE': 'str',
        'URBANIZEDAREA': 'str',
        'FUNCTIONALCLASSIFICATION': 'int64',
        'FEDERALFUNCTIONALCLASS': 'int64',
        'JURISDICTION': 'str',
        'TRUCKROUTE': 'str',
        'NHSSTATUS': 'str',  # National Highway System Status
        'FEDERALAIDROADNUMBER': 'str',
        'FACILITYTYPE': 'str',
        'STREETOPERATION': 'str',
        'ACCESSCONTROL': 'str',
        'TOLLROAD': 'str',
        
        # Traffic data
        'AADT': 'int64',  # Annual Average Daily Traffic
        'AADT_YEAR': 'int64',
        'AADT_DERIVATION': 'str',
        'STRUCTURALCONDITION': 'str',
        'SPEEDLIMIT': 'int64',
        'OPPOSINGDIRECTIONID': 'str',
        
        # Physical characteristics
        'NUMBEROFPEAKLANES': 'int64',
        'SURFACETYPE': 'str',
        'SURFACEWIDTH': 'float64',
        'RIGHTSIDEWALKWIDTH': 'float64',
        'RIGHTSHOULDERTYPE': 'str',
        'RIGHTSHOULDERWIDTH': 'float64',
        'MEDIANTYPE': 'str',
        'MEDIANWIDTH': 'float64',
        'LEFTSIDEWALKWIDTH': 'float64',
        'LEFTSHOULDERTYPE': 'str',
        'LEFTSHOULDERWIDTH': 'float64',
        'UNDIVIDEDLEFTSHOULDERWIDTH': 'float64',
        'UNDIVIDEDRIGHTSHOULDERWIDTH': 'float64',
        'CURBS': 'str',
        'TERRAIN': 'str',
        
        # Pavement condition fields  
        'IRI': 'float64',  # International Roughness Index
        'IRI_DATE': 'str',
        'PSI': 'float64',  # Present Serviceability Index
        'PCI': 'float64',  # Pavement Condition Index
        'PCI_DATE': 'str',
        
        # Additional boolean/flag fields (stored as integers)
        'NONSTATEOWNEDFEDAID': 'int64',
        'ADDEDMASSDOTMILES': 'int64',
        'DATECREATED': 'str',
        'CREATEDBY': 'str',
        'DATEUPDATED': 'str',
        'UPDATEDBY': 'str',
        'SHAPE_LENGTH': 'float64'
    }
    
    # Combine all field definitions
    all_fields = {**massgis_fields, **massdot_fields}
    
    # Create a mapping for common field name variations
    field_variations = {
        # Handle different naming conventions
        'SHAPE_LEN': 'float64',
        'SHAPE_Length': 'float64',
        'Shape_Length': 'float64',
        'FID': 'int64',
        'OBJECTID_1': 'int64',
        'Id': 'int64',
        'id': 'int64'
    }
    
    # Add field variations
    all_fields.update(field_variations)
    
    return all_fields

def apply_dtypes_to_geojson(filepath, output_path=None):
    """
    Apply the correct data types to a MassDOT Roads GeoJSON file.
    
    Parameters:
    -----------
    filepath : str or Path
        Path to the input GeoJSON file
    output_path : str or Path, optional
        Path for the output file. If None, adds '_typed' to the filename
    
    Returns:
    --------
    geopandas.GeoDataFrame
        The GeoDataFrame with corrected data types
    """
    # Get the data type definitions
    dtype_dict = set_massdot_roads_dtypes()
    
    # Read the GeoJSON file
    print(f"Reading GeoJSON file: {filepath}")
    gdf = gpd.read_file(filepath)
    
    # Apply data types to existing columns
    print("\nApplying data types to columns...")
    for col in gdf.columns:
        if col in dtype_dict and col != 'geometry':
            try:
                # Handle string type
                if dtype_dict[col] == 'str':
                    gdf[col] = gdf[col].astype(str).replace('None', '')
                # Handle integer type with null values
                elif dtype_dict[col] == 'int64':
                    gdf[col] = pd.to_numeric(gdf[col], errors='coerce').fillna(-999).astype('int64')
                # Handle float type
                elif dtype_dict[col] == 'float64':
                    gdf[col] = pd.to_numeric(gdf[col], errors='coerce')
                
                print(f"  - {col}: {gdf[col].dtype} (converted to {dtype_dict[col]})")
            except Exception as e:
                print(f"  - Warning: Could not convert {col} to {dtype_dict[col]}: {e}")
    
    # Save the updated GeoJSON
    if output_path is None:
        output_path = Path(filepath).parent / f"{Path(filepath).stem}_typed.geojson"
    
    print(f"\nSaving typed GeoJSON to: {output_path}")
    gdf.to_file(output_path, driver='GeoJSON')
    
    return gdf

def export_dtype_json(output_path='massdot_roads_dtypes.json'):
    """
    Export the data type definitions to a JSON file.
    
    Parameters:
    -----------
    output_path : str
        Path for the output JSON file
    """
    dtype_dict = set_massdot_roads_dtypes()
    
    # Create a more detailed JSON with field descriptions
    detailed_dtypes = {
        "dataset": "MassDOT Roads GeoJSON",
        "source": "Massachusetts Department of Transportation (MassDOT) and MassGIS",
        "fields": {}
    }
    
    # Add field descriptions
    field_descriptions = {
        'OBJECTID': {'type': 'int64', 'description': 'Unique object identifier'},
        'CLASS': {'type': 'int64', 'description': 'Road classification (1-7)'},
        'ADMIN_TYPE': {'type': 'int64', 'description': '0=Not applicable, 1=Interstate, 2=US Highway, 3=State Highway'},
        'STREET_NAME': {'type': 'str', 'description': 'Street name'},
        'RT_NUMBER': {'type': 'str', 'description': 'Route number'},
        'AADT': {'type': 'int64', 'description': 'Annual Average Daily Traffic'},
        'SPEEDLIMIT': {'type': 'int64', 'description': 'Posted speed limit'},
        'SURFACEWIDTH': {'type': 'float64', 'description': 'Surface width in feet'},
        'LENGTH_MI': {'type': 'float64', 'description': 'Segment length in miles'},
        'CITY': {'type': 'str', 'description': 'City name'},
        'FUNCTIONALCLASSIFICATION': {'type': 'int64', 'description': 'Functional classification code'},
        # Add more descriptions as needed
    }
    
    # Merge descriptions with data types
    for field, dtype in dtype_dict.items():
        if field in field_descriptions:
            detailed_dtypes['fields'][field] = field_descriptions[field]
        else:
            detailed_dtypes['fields'][field] = {'type': dtype, 'description': 'MassDOT field'}
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(detailed_dtypes, f, indent=2)
    
    print(f"Data type definitions exported to: {output_path}")
    
    return detailed_dtypes

if __name__ == "__main__":
    # Example usage
    input_file = "/Users/tommaso/prototypescorings/MassDOTRoads_gdb_1226590767708312459.geojson"
    
    # Apply data types to the GeoJSON file
    # gdf = apply_dtypes_to_geojson(input_file)
    
    # Export data type definitions to JSON
    dtype_json = export_dtype_json()
    
    print("\nData type setting complete!")
    print(f"Total fields defined: {len(dtype_json['fields'])}")
