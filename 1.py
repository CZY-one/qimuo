import streamlit as st
import pickle
import pandas as pd
def preprocess_inputs(age, sex, bmi, children, smoke, region):
    # 初始化变量
    sex_female, sex_male = 0, 0
    smoke_yes, smoke_no = 0, 0
    region_northeast, region_southeast, region_northwest, region_southwest = 0, 0, 0, 0
    
    # 处理性别
    if sex == '女性':
        sex_female = 1
    else:
        sex_male = 1
    
    # 处理吸烟情况
    if smoke == '是':
        smoke_yes = 1
    else:
        smoke_no = 1
    
    # 处理区域
    if region == '东北部':
        region_northeast = 1
    elif region == '东南部':
        region_southeast = 1
    elif region == '西北部':
        region_northwest = 1
    elif region == '西南部':
        region_southwest = 1
    
    return [age, bmi, children, sex_female, sex_male, 
            smoke_no, smoke_yes, 
            region_northeast, region_southeast, region_northwest, region_southwest]
def introduce_page():
    """当选择简介页面时，将呈现该函数的内容"""
    st.write("# 欢迎使用！")
    st.sidebar.success("单击 ⏭ 预测医疗费用")
def predict_page():
    """当选择预测费用页面时，将呈现该函数的内容"""
   
    
    # 使用唯一的表单键
    with st.form('prediction_form'):
        age = st.number_input('年龄', min_value=0)
        sex = st.radio('性别', options=['男性', '女性'])
        bmi = st.number_input('BMI', min_value=0.0)
        children = st.number_input("子女数量：", step=1, min_value=0)
        smoke = st.radio("是否吸烟", ("是", "否"))
        region = st.selectbox('区域', ('东南部', '西南部', '东北部', '西北部'))
        submitted = st.form_submit_button('预测费用')
    
    if submitted:
        # 预处理用户输入
        format_data = preprocess_inputs(age, sex, bmi, children, smoke, region)
        
        # 显示预处理后的数据
        st.write('转换为数据预处理的格式：')
        st.text(format_data)
        
        # 加载模型并预测
        try:
            with open('rfr_model.pkl', 'rb') as f:
                rfr_model = pickle.load(f)
            
            # 创建特征 DataFrame
            format_data_df = pd.DataFrame(data=[format_data], columns=rfr_model.feature_names_in_)
            
            # 预测并显示结果
            predict_result = rfr_model.predict(format_data_df)[0]
            st.write('根据您输入的数据，预测该客户的医疗费用是：', round(predict_result, 2))
        except Exception as e:
            st.error(f"模型加载或预测过程中出错: {str(e)}")
# 设置页面配置
st.set_page_config(
    page_title="医疗费用预测",
    page_icon="💰",
)
# 侧边栏导航
nav = st.sidebar.radio("导航", ["简介", "预测医疗费用"])
# 根据选择显示不同页面
if nav == "简介":
    introduce_page()
    st.markdown(
        """
        ### 使用说明
        这个应用利用机器学习模型来预测医疗费用，为保险公司的保险定价提供参考。
        - **输入信息**：在下面输入被保险人的个人信息、疾病信息等。
        - **费用预测**：应用会预测被保险人的未来医疗费用支出。
        """
    )
else:
    predict_page()

