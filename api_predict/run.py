import os
import json
import httpx

FEATURE_EXTRACTION_URL = "https://aiservice.tsghaiot.com/PASTER/api/v1/feature-extraction/"
LINEAR_PROBE_URL = "https://aiservice.tsghaiot.com/PASTER/api/v1/linear-probe/"


def response_print(response, if_print=False, print_content='not printed'):
    print(f"""
          request: {response.url}
          status_code: {response.status_code}
          content: {response.json() if if_print else print_content}
          """)
    return response.json()


def run(image_path=None):
    SSL_VERIFY = False
    response = httpx.get(FEATURE_EXTRACTION_URL, verify=SSL_VERIFY)
    content = response_print(response, True)

    response = httpx.get(LINEAR_PROBE_URL, verify=SSL_VERIFY)
    content = response_print(response, True)

    params = {"name": "USERNAME"}
    # filepath = "./cxr_test.png"

    files = {"file": open(image_path, 'rb')}

    response = httpx.post(
        f"{FEATURE_EXTRACTION_URL}cxr/image",
        params=params,
        files=files,
        timeout=15,
        verify=SSL_VERIFY
    )
    content = response_print(response, False, "save as json file")
    response_data = content
    # print(response_data['user'])
    feature = response_data['extracted_feature']
    with open('./feature.json', 'w') as f:
        f.write(json.dumps(response_data))

    # print(len(feature))
    if len(feature) == 1:
        feature = feature[0]

    data = feature
    response = httpx.post(
        f"{LINEAR_PROBE_URL}models/all",
        data=json.dumps(data),
        timeout=15,
        verify=SSL_VERIFY
    )
    content = response_print(response, True)


if __name__ == "__main__":
    run(image_path='./cxr_test.png')
