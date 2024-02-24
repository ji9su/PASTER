import os
import json
import httpx

FEATURE_EXTRACTION_URL = "https://aiservice.tsghaiot.com/PASTER/api/v1/feature-extraction/"
LINEAR_PROBE_URL = "https://aiservice.tsghaiot.com/PASTER/api/v1/linear-probe/"

AUTH = {'username': 'guest', 'password': 'guest@random'}


def response_print(response, if_print=False, print_content='not printed'):
    # print(f"""
    # results -> {response.json() if if_print else print_content}
    #       """)
    print(f"""
          request: {response.url}
          status_code: {response.status_code}
          content: {response.json() if if_print else print_content}
          """)
    return response.json()


def run(image_path=None):
    SSL_VERIFY = False
    AUTH = ("guest", "guest@random")

    files = {"file": open(image_path, 'rb')}

    print(f""" Running feature extraction ....
        input : {image_path}
        """)

    response = httpx.post(
        f"{FEATURE_EXTRACTION_URL}cxr/image",
        files=files,
        timeout=15,
        auth=AUTH,
        verify=SSL_VERIFY
    )
    content = response_print(response, False, "save as json file")
    response_data = content
    # print(response_data['user'])
    feature = response_data
    print(f"shape of feature : {len(feature)}")
    with open('./feature.json', 'w') as f:
        f.write(json.dumps(response_data))

    print('Using extracted feature to get analyzed results ...')

    response = httpx.post(
        f"{LINEAR_PROBE_URL}models/all",
        data=json.dumps(feature),
        timeout=15,
        auth=AUTH,
        verify=SSL_VERIFY
    )
    content = response_print(response, True)
    print(len(content))


if __name__ == "__main__":
    run(image_path='./xr_test.png')
