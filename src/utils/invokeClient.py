from tiingo import TiingoClient

def getClient():
    config = {}
    config['session'] = True
    try:
        config['api_key'] = "TIINGO_API_KEY" 
    except: 
        print("Exception caught while accessing data from API")
    client = TiingoClient(config)
    print("Client Invoked Successfully!")
    return client
