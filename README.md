
# Installation

### Prerequisites
```
Python 3.7+
```

### Install
```
pip3 install -r requirements.txt
```

# Run
### 1. 파일 변환 (pcap -> pkl)
```
// 옵션
[-p PATH] 패킷 파일 경로 설정 (파일 or 디렉토리)
[-a PATH] pkl 파일 경로를 저장할 파일 (기존 파일 내용에 이어서 입력됨 (append))
[-s] pkl 파일 저장 여부 설정
[-r] dummy 패킷 삭제 (pcap 파일안에서 가장 많은 src, dst flow를 제외한 나머지 패킷 삭제)

// 명령어
$ python3 pkl_maker.py -p [pcap file or directory] -a [where to save pkl paths] [-s] [-r]

// example
$ python3 pkl_maker.py -p ../data/old/etc -a pos_test.txt -s -r
$ python3 pkl_maker.py -p ../data/dummy_test -a neg_test.txt -s
$ python3 pkl_maker.py -p ../data/dummy/test_file.pcap -a neg_test.txt -s
```

### 2. 데이터 분석
```
// 명령어
$ python3 data_analysis.py

// data_analysis.py 모듈의 main 부분에서 직접 경로 설정을 해주어야함
if __name__ == '__main__':
    with open('pos_test.txt', 'r') as f:
      ...

    with open('neg_test.txt', 'r') as f:
      ...
```

### 3. 학습
```
// 명령어
$ python3 classifier.py

// classifier.py 모듈의 get_data() 함수 부분에서 직접 경로 설정을 해주어야함
def get_data():
    with open('pos_test.txt', 'r') as f:
      ...

    with open('neg_test.txt', 'r') as f:
      ...

```

### 4. Prediction
```
// 명령어
$ python predict.py

// predict.py 모듈의 main 부분에서 weight path, cap_path, param_path 직접 설정
if __name__ == '__main__':
	...

    model.load_weights('saved_models/weights.hdf5')
    cap_path = '...../camera.pcap'

    ...

    norm_params = load_norm_params('norm_params.txt')
```
