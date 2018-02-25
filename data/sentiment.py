def sent(text):

    import http.client, urllib.request, urllib.parse, urllib.error, base64, json


    accessKey = "dc8cd943ae384a4a8f7086791eadd581"

    uri = 'eastus.api.cognitive.microsoft.com'

    path = '/text/analytics/v2.0/sentiment'

    headers = {'Ocp-Apim-Subscription-Key': accessKey}
    conn = http.client.HTTPSConnection(uri)

    documents = { 'documents': [
    { 'id': '1', 'language': 'en', 'text': text},
    ]}

    body = json.dumps(documents)
    conn.request("POST", path, body, headers)
    response = conn.getresponse()

    result = response.read().decode('utf-8')

    json_result = json.loads(result)

    score = json_result["documents"][0]["score"]

    return(score)
