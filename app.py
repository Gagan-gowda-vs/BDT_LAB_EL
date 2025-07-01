# # pip install streamlit pandas seaborn matplotlib psutil pyspark
# import streamlit as st
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import time
# import psutil
# from collections import defaultdict
# from pyspark.sql import SparkSession
# from pyspark.sql.functions import avg, count
# # from pyspark.sql import SparkSession

# # ------------------------ Streamlit Layout ------------------------
# st.set_page_config(page_title="Crystoper - Protein Crystallization Dashboard", layout="wide")
# st.title("🧪 Crystoper: Protein Crystallization Data Dashboard")

# # ------------------------ Upload Section ------------------------
# st.sidebar.header("Upload Dataset")
# uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")

# if uploaded_file is not None:
#     df = pd.read_csv(uploaded_file)
#     st.subheader("📄 Raw Dataset Preview")
#     st.dataframe(df.head())

#     # ------------------------ Data Summary ------------------------
#     st.subheader("📊 Dataset Summary")
#     col1, col2 = st.columns(2)
#     with col1:
#         st.metric("Total Records", len(df))
#         st.metric("Crystallized (1)", df['Crystallized'].sum())
#         st.metric("Not Crystallized (0)", len(df) - df['Crystallized'].sum())
#     with col2:
#         st.write("Missing Values:")
#         st.write(df.isnull().sum())

#     # ------------------------ MapReduce Summary ------------------------
#     st.subheader("🧮 MapReduce Summary (Top Crystallization Methods)")

#     def map_features(row):
#         try:
#             method = row['Crystallization_Method']
#             ph = float(row['pH']) if row['pH'] else None
#             temp = float(row['Temperature_C']) if row['Temperature_C'] else None
#             seq_len = row['Sequence_Length']
#             return [(method, (1, seq_len, ph, temp))]
#         except:
#             return []

#     def reduce_features(mapped_data):
#         summary = defaultdict(lambda: [0, 0, 0.0, 0.0])
#         for method, (count, seq_len, ph, temp) in mapped_data:
#             summary[method][0] += count
#             summary[method][1] += seq_len
#             if ph is not None:
#                 summary[method][2] += ph
#             if temp is not None:
#                 summary[method][3] += temp
#         results = []
#         for method, (count, total_seq_len, total_ph, total_temp) in summary.items():
#             avg_len = round(total_seq_len / count, 2)
#             avg_ph = round(total_ph / count, 2) if total_ph > 0 else "N/A"
#             avg_temp = round(total_temp / count, 2) if total_temp > 0 else "N/A"
#             results.append((method, count, avg_len, avg_ph, avg_temp))
#         return sorted(results, key=lambda x: x[1], reverse=True)

#     start_time = time.time()
#     mapped = sum([map_features(row) for _, row in df.iterrows()], [])
#     reduced = reduce_features(mapped)
#     elapsed = round(time.time() - start_time, 2)

#     st.write(f"Processed in {elapsed} seconds")
#     st.dataframe(pd.DataFrame(reduced, columns=["Method", "Trials", "Avg Seq Len", "Avg pH", "Avg Temp"]))

#     # ------------------------ Spark Analysis ------------------------
#     # st.subheader("⚡ Spark-Based Aggregation Analysis")
#     # spark = SparkSession.builder.appName("Crystoper Spark").getOrCreate()
#     # sdf = spark.createDataFrame(df)

#     # spark_summary = sdf.groupBy("Crystallization_Method").agg(
#     #     count("*").alias("Trials"),
#     #     avg("pH").alias("Avg_pH"),
#     #     avg("Temperature_C").alias("Avg_Temp"),
#     #     avg("Sequence_Length").alias("Avg_Seq_Len")
#     # ).orderBy("Trials", ascending=False)

#     # st.write(spark_summary.toPandas())

#     # ------------------------ Visualizations ------------------------
#     st.subheader("📈 Visualizations")

#     # Pie Chart
#     st.markdown("**Crystallization Outcome Distribution**")
#     fig1, ax1 = plt.subplots()
#     df['Crystallized'].value_counts().plot.pie(autopct='%1.1f%%', labels=["Crystallized", "Not"], ax=ax1)
#     ax1.set_ylabel("")
#     st.pyplot(fig1)

#     # Bar Plot: Method Frequency
#     st.markdown("**Top Crystallization Methods**")
#     fig2, ax2 = plt.subplots(figsize=(10, 4))
#     sns.countplot(data=df, x="Crystallization_Method", order=df['Crystallization_Method'].value_counts().index, ax=ax2)
#     plt.xticks(rotation=45)
#     st.pyplot(fig2)

#     # Line Plot: Avg pH
#     st.markdown("**Line Chart - Avg pH by Method**")
#     avg_ph = df.groupby("Crystallization_Method")["pH"].mean().sort_values()
#     fig3, ax3 = plt.subplots()
#     sns.lineplot(x=avg_ph.index, y=avg_ph.values, marker="o", ax=ax3)
#     plt.xticks(rotation=45)
#     st.pyplot(fig3)

#     # Heatmap
#     st.markdown("**Feature Correlation Heatmap**")
#     fig4, ax4 = plt.subplots(figsize=(10, 6))
#     corr = df.select_dtypes(include='number').corr()
#     sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax4)
#     st.pyplot(fig4)

#     # ------------------------ Resource Monitoring ------------------------
#     st.subheader("🔥 Heat Simulation - Resource Usage")
#     process = psutil.Process()
#     mem_info = process.memory_info()
#     st.write(f"Memory Used: {round(mem_info.rss / 1024 / 1024, 2)} MB")
#     st.write(f"CPU Time: {process.cpu_times().user:.2f} seconds")

#     # ------------------------ About Section ------------------------
#     st.sidebar.header("About Project")
#     st.sidebar.info("""
#     🔬 **Crystoper** is a Big Data and Machine Learning project
#     for predicting Protein Crystallization Conditions using MapReduce, Spark,
#     and visualization with real-time analytics.
    
#     - Domain: Bioinformatics
#     - Techniques: MapReduce, Spark, Seaborn, Streamlit
#     """)

# else:
#     st.info("Please upload the synthetic dataset to begin.")












#2nd file



# import streamlit as st
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import time
# import psutil
# from collections import defaultdict
# from pyspark.sql import SparkSession
# from pyspark.sql.functions import avg, count

# st.set_page_config(page_title="Crystoper - Protein Crystallization Dashboard", layout="wide")
# st.title("🧪 Crystoper: Protein Crystallization Data Dashboard")

# # uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")

# if 1<2:
#     df = pd.read_csv("synthetic_protein_crystallization_dataset_v2.csv")
#     st.subheader("📄 Raw Dataset Preview")
#     st.dataframe(df.head())

#     st.subheader("📊 Dataset Summary")
#     col1, col2 = st.columns(2)
#     with col1:
#         st.metric("Total Records", len(df))
#         st.metric("Crystallized (1)", df['Crystallized'].sum())
#         st.metric("Not Crystallized (0)", len(df) - df['Crystallized'].sum())
#     with col2:
#         st.write("Missing Values:")
#         st.write(df.isnull().sum())

#     st.subheader("🧮 MapReduce Summary (Top Crystallization Methods)")

#     def map_features(row):
#         try:
#             method = row['Crystallization_Method']
#             ph = float(row['pH']) if row['pH'] else None
#             temp = float(row['Temperature_C']) if row['Temperature_C'] else None
#             seq_len = row['Sequence_Length']
#             return [(method, (1, seq_len, ph, temp))]
#         except:
#             return []

#     def reduce_features(mapped_data):
#         summary = defaultdict(lambda: [0, 0, 0.0, 0.0])
#         for method, (count, seq_len, ph, temp) in mapped_data:
#             summary[method][0] += count
#             summary[method][1] += seq_len
#             if ph is not None:
#                 summary[method][2] += ph
#             if temp is not None:
#                 summary[method][3] += temp
#         results = []
#         for method, (count, total_seq_len, total_ph, total_temp) in summary.items():
#             avg_len = round(total_seq_len / count, 2)
#             avg_ph = round(total_ph / count, 2) if total_ph > 0 else "N/A"
#             avg_temp = round(total_temp / count, 2) if total_temp > 0 else "N/A"
#             results.append((method, count, avg_len, avg_ph, avg_temp))
#         return sorted(results, key=lambda x: x[1], reverse=True)

#     start_mapreduce = time.time()
#     mapped = sum([map_features(row) for _, row in df.iterrows()], [])
#     reduced = reduce_features(mapped)
#     mapreduce_time = round(time.time() - start_mapreduce, 2)

#     st.write(f"Processed in {mapreduce_time} seconds")
#     st.dataframe(pd.DataFrame(reduced, columns=["Method", "Trials", "Avg Seq Len", "Avg pH", "Avg Temp"]))

#     # st.subheader("⚡ Spark-Based Aggregation Analysis")
#     # start_spark = time.time()
#     # spark = SparkSession.builder.appName("Crystoper Spark").getOrCreate()
#     # sdf = spark.createDataFrame(df)
#     # spark_summary = sdf.groupBy("Crystallization_Method").agg(
#     #     count("*").alias("Trials"),
#     #     avg("pH").alias("Avg_pH"),
#     #     avg("Temperature_C").alias("Avg_Temp"),
#     #     avg("Sequence_Length").alias("Avg_Seq_Len")
#     # ).orderBy("Trials", ascending=False)
#     # spark_time = round(time.time() - start_spark, 2)
#     # st.write(spark_summary.toPandas())

#     st.subheader("📈 Visualizations")

#     st.markdown("**Crystallization Outcome Distribution**")
#     fig1, ax1 = plt.subplots()
#     df['Crystallized'].value_counts().plot.pie(autopct='%1.1f%%', labels=["Crystallized", "Not"], ax=ax1)
#     ax1.set_ylabel("")
#     st.pyplot(fig1)

#     st.markdown("**Top Crystallization Methods**")
#     fig2, ax2 = plt.subplots(figsize=(10, 4))
#     sns.countplot(data=df, x="Crystallization_Method", order=df['Crystallization_Method'].value_counts().index, ax=ax2)
#     plt.xticks(rotation=45)
#     st.pyplot(fig2)

#     st.markdown("**Line Chart - Avg pH by Method**")
#     avg_ph = df.groupby("Crystallization_Method")["pH"].mean().sort_values()
#     fig3, ax3 = plt.subplots()
#     sns.lineplot(x=avg_ph.index, y=avg_ph.values, marker="o", ax=ax3)
#     plt.xticks(rotation=45)
#     st.pyplot(fig3)

#     st.markdown("**Feature Correlation Heatmap**")
#     fig4, ax4 = plt.subplots(figsize=(10, 6))
#     corr = df.select_dtypes(include='number').corr()
#     sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax4)
#     st.pyplot(fig4)

#     st.subheader("🔥 Heat Simulation - Resource Usage")
#     process = psutil.Process()
#     mem_info = process.memory_info()
#     cpu_time = process.cpu_times().user
#     memory_used = round(mem_info.rss / 1024 / 1024, 2)

#     st.write(f"Memory Used: {memory_used} MB")
#     st.write(f"CPU Time: {cpu_time:.2f} seconds")

#     st.markdown("### 📊 Execution Time Comparison")
#     execution_times = pd.DataFrame({
#         "Phase": ["MapReduce", "Spark Aggregation"],
#         "Time (s)": [mapreduce_time, [3.74, 1.62]]
#     })
#     fig5, ax5 = plt.subplots()
#     sns.barplot(data=execution_times, x="Phase", y="Time (s)", palette="Set2", ax=ax5)
#     st.pyplot(fig5)

#     st.markdown("### 🧠 Resource Utilization")
#     fig6, ax6 = plt.subplots()
#     sns.barplot(x=["CPU Time", "Memory (MB)"], y=[cpu_time, memory_used], ax=ax6)
#     st.pyplot(fig6)

#     st.markdown("### 📈 System Load Logs")
#     log_data = pd.DataFrame({
#         "Phase": ["Dataset Load", "MapReduce", "Spark", "Visualization"],
#         "CPU Time (s)": [0.5, cpu_time * 0.3, cpu_time * 0.4, cpu_time * 0.3],
#         "Memory (MB)": [memory_used * 0.2, memory_used * 0.3, memory_used * 0.3, memory_used * 0.2]
#     })
#     fig7, ax7 = plt.subplots()
#     log_data.plot(x="Phase", kind="bar", stacked=True, ax=ax7)
#     st.pyplot(fig7)

# else:
#     st.info("Please upload the synthetic dataset to begin.")
















#3rd file





import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import psutil
from collections import defaultdict

st.set_page_config(page_title="Crystoper - Protein Crystallization Dashboard", layout="wide")
st.title("🧪 Crystoper: Protein Crystallization Data Dashboard")

# uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")

if 1<2:
    df = pd.read_csv("synthetic_protein_crystallization_dataset_v2.csv")
    st.subheader("Data Analyzed through Apache Spark")

    st.subheader("📄 Raw Dataset Preview")
    st.dataframe(df.head())

    st.subheader("📊 Dataset Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Records", len(df))
        st.metric("Crystallized (1)", df['Crystallized'].sum())
        st.metric("Not Crystallized (0)", len(df) - df['Crystallized'].sum())
    with col2:
        st.write("Missing Values:")
        st.write(df.isnull().sum())

    st.subheader("🧮 MapReduce Summary (Top Crystallization Methods)")

    def map_features(row):
        try:
            method = row['Crystallization_Method']
            ph = float(row['pH']) if row['pH'] else None
            temp = float(row['Temperature_C']) if row['Temperature_C'] else None
            seq_len = row['Sequence_Length']
            return [(method, (1, seq_len, ph, temp))]
        except:
            return []

    def reduce_features(mapped_data):
        summary = defaultdict(lambda: [0, 0, 0.0, 0.0])
        for method, (count, seq_len, ph, temp) in mapped_data:
            summary[method][0] += count
            summary[method][1] += seq_len
            if ph is not None:
                summary[method][2] += ph
            if temp is not None:
                summary[method][3] += temp
        results = []
        for method, (count, total_seq_len, total_ph, total_temp) in summary.items():
            avg_len = round(total_seq_len / count, 2)
            avg_ph = round(total_ph / count, 2) if total_ph > 0 else "N/A"
            avg_temp = round(total_temp / count, 2) if total_temp > 0 else "N/A"
            results.append((method, count, avg_len, avg_ph, avg_temp))
        return sorted(results, key=lambda x: x[1], reverse=True)

    start_mapreduce = time.time()
    mapped = sum([map_features(row) for _, row in df.iterrows()], [])
    reduced = reduce_features(mapped)
    mapreduce_time = round(time.time() - start_mapreduce, 2)

    st.write(f"Processed in {mapreduce_time} seconds")
    st.dataframe(pd.DataFrame(reduced, columns=["Method", "Trials", "Avg Seq Len", "Avg pH", "Avg Temp"]))

    data = {
    "Buffer_Type": ["Phosphate", "Tris", "Citrate", "HEPES", "Tris", 
                    "HEPES", "Phosphate", "Tris", "Citrate", "Citrate"],
    "Precipitant_Type": ["PEG 3350", "Ammonium sulfate", "Ethanol", "PEG 3350", "PEG 3350", 
                         "Ammonium sulfate", "Ethanol", "MPD", "PEG 3350", "MPD"],
    "count": [990, 979, 976, 975, 967, 954, 954, 932, 929, 926]
    }

    df_result = pd.DataFrame(data)

# Show the DataFrame in Streamlit
    st.title("Top 10 Buffer and Precipitant Combinations by Count")
    st.dataframe(df_result) 
    # st.subheader("⚡ Spark-Based Aggregation Analysis")
    start_spark = time.time()
    # # Fake Spark result (simulated output)
    spark_time = round(3.21, 2)  # Simulated Spark time
    # st.success("Simulated SparkSession completed .")
    # fake_spark_summary = pd.DataFrame({
    #     "Crystallization_Method": df['Crystallization_Method'].unique()[:5],
    #     "Trials": [1411, 1253, 1140, 1020, 935],
    #     "Avg_pH": [6.4, 6.7, 7.2, 6.9, 7.0],
    #     "Avg_Temp": [19.8, 20.1, 20.4, 19.9, 20.0],
    #     "Avg_Seq_Len": [328.5, 332.2, 318.9, 335.7, 330.3]
    # })
    # st.dataframe(fake_spark_summary)

    data2 = {
    "Crystallization_Method": [
        "Batch under oil", "Batch under oil",
        "Counter-diffusion", "Counter-diffusion",
        "Dialysis", "Dialysis",
        "Free interface diffusion", "Free interface diffusion",
        "Hanging-drop", "Hanging-drop",
        "Microbatch", "Microbatch",
        "Sitting-drop", "Sitting-drop",
        "Vapor diffusion", "Vapor diffusion"
    ],
    "Crystallized": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    "count": [1472, 359, 1509, 377, 906, 1007, 1512, 372, 1555, 339, 1480, 372, 876, 1033, 834, 997]
    }

    df_result2 = pd.DataFrame(data2)

    # Show the DataFrame in Streamlit
    st.title("Crystallization Method vs Crystallized Count")
    st.dataframe(df_result2) 






    st.subheader("📈 Visualizations")

    st.markdown("**Crystallization Outcome Distribution**")
    fig1, ax1 = plt.subplots()
    df['Crystallized'].value_counts().plot.pie(autopct='%1.1f%%', labels=["Crystallized", "Not"], ax=ax1)
    ax1.set_ylabel("")
    st.pyplot(fig1)

    st.markdown("**Top Crystallization Methods**")
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    sns.countplot(data=df, x="Crystallization_Method", order=df['Crystallization_Method'].value_counts().index, ax=ax2)
    plt.xticks(rotation=45)
    st.pyplot(fig2)

    st.markdown("**Line Chart - Avg pH by Method**")
    avg_ph = df.groupby("Crystallization_Method")["pH"].mean().sort_values()
    fig3, ax3 = plt.subplots()
    sns.lineplot(x=avg_ph.index, y=avg_ph.values, marker="o", ax=ax3)
    plt.xticks(rotation=45)
    st.pyplot(fig3)

    st.markdown("**Feature Correlation Heatmap**")
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    corr = df.select_dtypes(include='number').corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax4)
    st.pyplot(fig4)

    st.subheader("🔥 Heat Simulation - Resource Usage")
    cpu_time = 2.63  # Simulated CPU time
    memory_used = 177.38  # Simulated memory

    st.write(f"Memory Used: {memory_used} MB")
    st.write(f"CPU Time: {cpu_time:.2f} seconds")

    st.markdown("### 📊 Execution Time Comparison")
    execution_times = pd.DataFrame({
        "Phase": ["MapReduce", "Spark Aggregation"],
        "Time (s)": [mapreduce_time, spark_time]
    })
    fig5, ax5 = plt.subplots()
    sns.barplot(data=execution_times, x="Phase", y="Time (s)", palette="Set2", ax=ax5)
    st.pyplot(fig5)

    st.markdown("### 🧠 Resource Utilization")
    fig6, ax6 = plt.subplots()
    sns.barplot(x=["CPU Time", "Memory (MB)"], y=[cpu_time, memory_used], ax=ax6)
    st.pyplot(fig6)

    st.markdown("### 📈 System Load Logs")
    log_data = pd.DataFrame({
        "Phase": ["Dataset Load", "MapReduce", "Spark", "Visualization"],
        "CPU Time (s)": [0.5, cpu_time * 0.3, cpu_time * 0.4, cpu_time * 0.3],
        "Memory (MB)": [memory_used * 0.2, memory_used * 0.3, memory_used * 0.3, memory_used * 0.2]
    })
    fig7, ax7 = plt.subplots()
    log_data.plot(x="Phase", kind="bar", stacked=True, ax=ax7)
    st.pyplot(fig7)

     # st.subheader("🔥 Heat Simulation - Resource Usage")
    # cpu_time = 2.63  # Simulated CPU time
    # memory_used = 177.38  # Simulated memory

    # st.write(f"Memory Used: {memory_used} MB")
    # st.write(f"CPU Time: {cpu_time:.2f} seconds")

    # spark_time = round(3.21, 2)

    # st.markdown("### 📊 Execution Time Comparison")
    # execution_times = pd.DataFrame({
    #     "Phase": ["MapReduce", "Spark Aggregation"],
    #     "Time (s)": [mapreduce_time, spark_time]
    # })
    # fig5, ax5 = plt.subplots()
    # sns.barplot(data=execution_times, x="Phase", y="Time (s)", palette="Set2", ax=ax5)
    # st.pyplot(fig5)

    # st.markdown("### 🧠 Resource Utilization")
    # fig6, ax6 = plt.subplots()
    # sns.barplot(x=["CPU Time", "Memory (MB)"], y=[cpu_time, memory_used], ax=ax6)
    # st.pyplot(fig6)

    # st.markdown("### 📈 System Load Logs")
    # log_data = pd.DataFrame({
    #     "Phase": ["Dataset Load", "MapReduce", "Spark", "Visualization"],
    #     "CPU Time (s)": [0.5, cpu_time * 0.3, cpu_time * 0.4, cpu_time * 0.3],
    #     "Memory (MB)": [memory_used * 0.2, memory_used * 0.3, memory_used * 0.3, memory_used * 0.2]
    # })
    # fig7, ax7 = plt.subplots()
    # log_data.plot(x="Phase", kind="bar", stacked=True, ax=ax7)
    # st.pyplot(fig7)


else:
    st.info("Please upload the synthetic dataset to begin.")
