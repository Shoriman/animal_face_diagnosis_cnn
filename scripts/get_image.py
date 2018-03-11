# -*- coding: utf-8 -*-
import argparse
import requests
import urllib.parse
import io
import imghdr

# 基本的なモデルパラメータ
FLAGS = None

# エンドポイント
kEndPoint = 'https://api.cognitive.microsoft.com/bing/v7.0/images/search'

# http リクエストヘッダ
kHeaders = {'Ocp-Apim-Subscription-Key': 'e34b64913c71407fa160662c231d45ec'}


# 検索結果の画像URL一覧を取得
def GetImageUrls():
    print('Start getting %d images from offset %d' % (FLAGS.image_count,
                                                      FLAGS.off_set_start))
    image_list = []

    # bing APIの制限で150件までなので、ループしてcall_countの回数分取得
    for step in range(FLAGS.call_count):

        # 取得オフセット
        off_set = FLAGS.off_set_start + step * FLAGS.image_count

        # httpリクエストのパラメータ
        params = urllib.parse.urlencode({
            'count': FLAGS.image_count,
            'offset': off_set,
            'imageType': 'Photo',
            'q': FLAGS.query,
        })

        # bing API呼出
        res = requests.get(kEndPoint, headers=kHeaders, params=params)

        if step == 0:
            print('Total Estimated Matches: %s' %
                  res.json()['totalEstimatedMatches'])
        vals = res.json()['value']

        print('Get %d images from offset %d' % (len(vals), off_set))

        # 結果の画像URLを格納
        for j in range(len(vals)):
            image_list.append(vals[j]["contentUrl"])

    return image_list


# 画像を取得してローカルへ保存
def fetch_images(image_list):
    print('total images:%d' % len(image_list))
    for i in range(len(image_list)):

        # 100件ごとに進捗出力
        if i % 100 == 0:
            print('Start getting and saving each image:%d' % i)
        try:
            # 画像取得
            response = requests.get(image_list[i], timeout=5)

        # 取得元によってエラーが起きる場合あるため、ログだけ出しておいて続行
        except requests.exceptions.RequestException:
            print('%d:Error occurs :%s' % (i, image_list[i]))
            continue

        # 画像種類でフィルタ
        with io.BytesIO(response.content) as fh:
            if imghdr.what(fh) != 'jpeg' and imghdr.what(fh) != 'png' and\
               imghdr.what(fh) != 'tiff':
                print('Not saved file type:%s' % imghdr.what(fh))
                continue

        # 画像をローカル保存
            with open('{}/image_{}.{}'.format(FLAGS.output_path,
                                              str(i), imghdr.what(fh)),
                      'wb') as f:
                f.write(response.content)


# 直接実行されている場合に通る(importされて実行時は通らない)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_count',
        type=int,
        default=150,
        help='collection number of image files per api call.'
    )
    parser.add_argument(
        '--call_count',
        type=int,
        default=10,
        help='number of api calls.'
    )
    parser.add_argument(
        '--off_set_start',
        type=int,
        default=0,
        help='offset start.'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default='./images',
        help='image files output directry.'
    )
    parser.add_argument(
        '--query',
        type=str,
        default='猫',
        help='search query.'
    )

    # パラメータ取得と実行
    FLAGS, unparsed = parser.parse_known_args()
    fetch_images(GetImageUrls())
