from tiingo import TiingoClient

def getClient():
    config = {}
    config['session'] = True
    try:
        config['api_key'] = "44c3cbbc4115125fe0f7012c67b29d10f0be5a00" 
    except: 
        print("Exception caught while accessing data from API")
    client = TiingoClient(config)
    print("Client Invoked Successfully!")
    return client