from langchain_core.tools import tool

@tool
def calculate_bmi(height, weight):
  """BMIを計算する関数

  Args:
    height: 身長 (m)
    weight: 体重 (kg)

  Returns:
    BMI: 体重指数
  """

  bmi = weight / (height ** 2)
  return bmi

