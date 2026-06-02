# 문제 1. OpenAI API를 이용한 학습 설명문 생성 〔50점〕
# 작업 상황
# OpenAI API를 호출하여 **고등학생을 대상으로 한 “인공지능 개념 설명문”**을 생성하시오.

# 단순히 ChatGPT 화면에서 질문하는 것이 아니라, 직접 프롬프트를 작성하고 OpenAI API에 전달하여 응답을 받아야 한다.

# 작업 요구사항
# 다음 조건을 만족하는 프롬프트를 작성하고, OpenAI API를 호출하여 응답을 생성하시오.

# 생성 주제
# 인공지능의 개념과 생활 속 활용 사례

# 프롬프트 작성 조건
# 프롬프트에는 다음 조건이 반드시 포함되어야 한다.

# 1. AI의 역할: 고등학생에게 쉽게 설명하는 정보 교사 2. 대상: 고등학생 3. 주제: 인공지능의 개념과 생활 속 활용 사례 4. 분량: 500자 내외 5. 어려운 전문 용어 최소화 6. 쉬운 예시 3개 이상 포함 7. 마지막에 핵심 요약 3줄 포함 8. 출력 형식 지정

# API 요청 구조 조건
# API 요청에는 다음 요소가 포함되어야 한다.

# model messages temperature max_tokens

# messages에는 최소한 다음 역할이 포함되어야 한다.

# system 메시지 user 메시지

# 제출 내용
# 다음 항목을 제출하시오.

# [문제 1 제출] 1. 작성한 프롬프트 2. OpenAI API 요청 구조 3. API 응답 결과 4. 응답 결과 검토 내용 5. 개선한 점 또는 느낀 점

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM = f"""너는 고등학생에게 AI에 대한 정보를 쉽게 설명하는 교사야. 
          아래의 5가지 규칙을 반드시 지켜줘
          1. 주제: 인공지능의 개념과 생활 속 활용 사례 
          2. 분량: 500자 내외 
          3. 어려운 전문 용어 최소화 
          4. 쉬운 예시 3개 이상 포함 
          5. 마지막에 핵심 요약 3줄 포함해줘. 
          
          반드시 아래 JSON형식으로만 응답하세요.
{{"topic" : "인공지능 개념 설명문", "aiConcept": "인공지능에 대한 개념 200자 이내", "useCases": ["쉬운 예시 1번", "쉬운 예시 2번", "쉬운 예시 3번"], "
summation" : "인공지능 개념 간단한 요약 3줄"}}
          """
USER = f"인공지능에 대해서 설명해줘"

response = client.chat.completions.create(
            model       = "gpt-4o",
            messages    = [
                {"role": "system", "content": SYSTEM},
                {"role": "user",   "content": USER},
            ],
            max_tokens  = 500,
            temperature = 1.3,
        )
answer = response.choices[0].message.content

print(type(answer))
print(answer)






# 문제 2. OpenAI API를 이용한 상품 리뷰 JSON 분석 〔50점〕
# 작업 상황
# OpenAI API를 호출하여 상품 리뷰 문장을 분석하고, 결과를 JSON 형식으로 응답받으시오.

# 이 문제의 핵심은 AI 응답을 프로그램에서 활용할 수 있도록 정해진 JSON 구조로 받는 것이다.

 

# 입력 리뷰
# 다음 상품 리뷰를 분석하시오.

# 배송은 빨랐지만 제품 마감이 조금 아쉬웠어요. 그래도 가격을 생각하면 나쁘지는 않습니다.

 

# 작업 요구사항
# OpenAI API를 사용하여 위 리뷰를 분석하고, 응답을 JSON 형식으로 받으시오.

# JSON에는 다음 항목이 반드시 포함되어야 한다.

# sentiment summary keywords score reason

 

# JSON 항목 조건
# 각 항목은 다음 조건을 만족해야 한다.

# 1. sentiment   - positive, negative, neutral 중 하나로 출력 
# 2. summary   - 리뷰 내용을 한 문장으로 요약 
# 3. keywords   - 핵심 키워드를 배열 형태로 출력 
# 4. score   - 만족도를 1~5 사이 숫자로 출력 
# 5. reason   - sentiment와 score를 판단한 이유 작성

 

# 프롬프트 작성 조건
# 프롬프트에는 다음 조건이 반드시 포함되어야 한다.

# 1. 상품 리뷰를 분석하라고 지시 2. JSON 형식으로만 출력하도록 지시 3. JSON 외 설명 문장 출력 금지 4. sentiment 값 제한 5. score는 1~5 사이 숫자로 제한 6. keywords는 배열 형태로 출력 7. 올바른 JSON 문법 사용 지시

 

# API 요청 구조 조건
# API 요청에는 다음 요소가 포함되어야 한다.

# model messages temperature max_tokens response_format 또는 JSON 출력 강제 지시

# 수업에서 response_format을 배운 경우 다음과 같은 JSON 응답 형식 옵션을 사용할 수 있다.

# response_format: { "type": "json_object" }

 

# 제출 내용
# 다음 항목을 제출하시오.

# [문제 2 제출] 1. 작성한 프롬프트 2. OpenAI API 요청 구조 3. API 응답 결과 4. 응답 결과 검토 내용 5. 개선한 점 또는 느낀 점

import json

review = "배송은 빨랐지만 제품 마감이 조금 아쉬웠어요. 그래도 가격을 생각하면 나쁘지는 않습니다."
USERPROMPT = f"""{review}
              위 리뷰를 분석해줘.
              """


SYSTEMTWO = f"""당신은 상품 리뷰 분석가입니다.
리뷰를 분석하여 반드시 아래 JSON 형식으로만 답하세요.
다른 텍스트는 절대 포함하지 마세요.


{{  
"sentiment": 전체 감성: positive, neutral, negative 중 하나, 
"summary": 한 줄 요약,
"keywords" : 핵심 키워드를 배열 형태로 출력,
"rating": 별점 예측: 1~5 사이 정수,
"reason" : sentiment와 score를 판단한 이유 작성
}}

예시 형식 :
{{  
"sentiment": "positive",
"summary": "배송과 배터리는 만족스럽지만 마감 품질은 아쉬운 리뷰입니다.",
"keywords" : ["배송이 빠름", "배터리가 오래감", "가성비"],
"rating": 4,
"reason": "전체적으로 상품에 대한 긍정적인 리뷰가 많았음."
}}"""

myFormat = {
    "type": "json_schema",
    "json_schema" : {
        "name" : "A_unique_answer",
        "schema" : {
            "type": "object",
            "properties": {
                "Info": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties" : {
                            "sentiment" : {
                                "type": "string",
                                "description" : "리뷰에 대한 감정 평가입니다. positive, neutral, negative 중 하나를 작성합니다."
                            },
                            "summary" : {
                                "type": "string",
                                "description" : "리뷰에 대한 한줄 요약입니다. 50자 이내로 작성합니다."
                            },
                            "keywords" : {
                                "type": "array",
                                "items" : 
                                      {
                                        "type": "string",
                                        "description" : "리뷰 평가 판단의 핵심적인 키워드를 리뷰안에서 가져와 작성합니다."
                                      }
                            },
                            "rating" : {
                                "type": "integer",
                                "description" : "별점을 1~5 범위의 정수로 작성합니다. 1에 가까울수록 부정적이고 5에 가까울수록 긍정적입니다."
                            },
                            "reason" : {
                                "type": "string",
                                "description" : "sentiment와 score를 판단한 이유 작성"
                            }
                        },
                        "additionalProperties": False,
                        "required" : ["sentiment", "summary", "keywords", "rating", "reason"]
                    }
                }
            },
            "additionalProperties": False,
            "required": ["Info"]
        },
        "strict" : True
    }
}

messages = [
	{
   "role":"system",
    "content": SYSTEMTWO
  },
    {
      "role": "user", 
      "content": USERPROMPT
      }
]

response2 = client.chat.completions.create(
		model="gpt-4o",
		messages=messages,
    response_format=myFormat,
    temperature=1.0,
    max_tokens=1000
)

if response2 is not None :
  answer = response2.choices[0].message.content
  print(answer)
  
  answer_dict = json.loads(answer)
  print(answer_dict['Info'])
  answer_list = answer_dict['Info']
  for answer in answer_list :
    print(answer['sentiment'])
    print(answer['summary'])
    for keyword in answer['keywords'] :
      print(keyword)
    print(answer['rating'])
    print(answer['reason'])
  
