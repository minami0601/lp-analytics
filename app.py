import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import io
from datetime import datetime
import numpy as np
from PIL import Image

# ページ設定
st.set_page_config(page_title="LPデータ分析", layout="wide")
st.title("LPデータ分析ダッシュボード")

# CSVデータの読み込み


@st.cache_data
def load_data():
    # CSVファイルを読み込む
    df = pd.read_csv('LPデータ.csv')

    # デバイスタイプを判別する関数
    def get_device_type(user_agent):
        if 'iPad' in user_agent:
            return 'タブレット'
        elif 'iPhone' in user_agent or 'Android' in user_agent:
            return 'スマートフォン'
        elif 'Macintosh' in user_agent or 'Windows' in user_agent:
            return 'PC'
        return 'その他'

    # デバイスタイプを追加
    df['デバイスタイプ'] = df['デバイス'].apply(get_device_type)

    # 日付をdatetime型に変換
    df['アクセス時間'] = pd.to_datetime(df['日付'])

    # 動画再生時間を秒から分に変換
    df['video_time_min'] = df['動画再生時間(秒)'] / 60

    # 各セッションの最後のアクションを取得
    last_actions = df.sort_values('アクセス時間').groupby(
        'session ID').last().reset_index()

    return last_actions


# データの読み込み
df = load_data()

# サイドバーにフィルター追加
st.sidebar.header("フィルター")
device_filter = st.sidebar.multiselect(
    "デバイスタイプ",
    options=['スマートフォン', 'タブレット', 'PC', 'その他'],
    default=['スマートフォン', 'タブレット', 'PC', 'その他']
)

guide_filter = st.sidebar.multiselect(
    "案内表示",
    options=df['案内が表示されたか'].unique(),
    default=df['案内が表示されたか'].unique()
)

# フィルタリングされたデータ
filtered_df = df[
    (df['デバイスタイプ'].isin(device_filter)) &
    (df['案内が表示されたか'].isin(guide_filter))
]

# 2カラムレイアウト
col1, col2 = st.columns(2)

with col1:
    st.subheader("動画視聴時間の分布")
    # 動画視聴時間のヒストグラム
    fig_video = px.histogram(
        filtered_df[filtered_df['video_time_min'] > 0],
        x='video_time_min',
        nbins=20,
        title="動画視聴時間の分布"
    )
    fig_video.update_layout(
        xaxis_title="視聴時間（分）",
        yaxis_title="視聴者数"
    )
    st.plotly_chart(fig_video, use_container_width=True)

    # 基本統計量
    st.subheader("動画視聴時間の基本統計量")
    video_stats = filtered_df[filtered_df['video_time_min']
                              > 0]['video_time_min'].describe().round(2)
    st.write(video_stats)

with col2:
    st.subheader("デバイス・案内表示別スクロール率")
    # デバイスと案内表示別の平均スクロール率
    scroll_avg = filtered_df.groupby(['デバイスタイプ', '案内が表示されたか'])[
        'スクロール率(%)'].mean().reset_index()
    fig_scroll = px.bar(
        scroll_avg,
        x='デバイスタイプ',
        y='スクロール率(%)',
        color='案内が表示されたか',
        barmode='group',
        title="デバイス・案内表示別の平均スクロール率",
        category_orders={"デバイスタイプ": ['スマートフォン', 'タブレット', 'PC', 'その他']}
    )
    fig_scroll.update_layout(
        xaxis_title="デバイスタイプ",
        yaxis_title="平均スクロール率（%）"
    )
    st.plotly_chart(fig_scroll, use_container_width=True)

    # スクロール率の基本統計量
    st.subheader("スクロール率の基本統計量")
    scroll_stats = filtered_df['スクロール率(%)'].describe()
    st.write(scroll_stats)

# スクロール率のヒートマップ
st.subheader("案内表示別スクロール率の分布ヒートマップ（最終アクション）")

# スクロール率を10%区切りの範囲に分類
bins = list(range(0, 101, 10))
labels = [f'{i}%-{i+10}%' for i in range(0, 100, 10)]

# 案内表示の有無でグループ化してヒートマップデータを作成
heatmap_data = []
y_labels = []

for guide in sorted(filtered_df['案内が表示されたか'].unique()):
    guide_df = filtered_df[filtered_df['案内が表示されたか'] == guide]
    guide_df['scroll_range'] = pd.cut(
        guide_df['スクロール率(%)'], bins=bins, labels=labels, include_lowest=True)

    # スクロール率範囲ごとの件数を計算
    scroll_distribution = guide_df['scroll_range'].value_counts().sort_index()
    total_count = len(guide_df)
    scroll_percentages = (scroll_distribution / total_count * 100).round(1)

    heatmap_data.append(scroll_percentages.values)
    y_labels.append(f'案内{guide}')

# ヒートマップの作成
fig_heatmap = go.Figure(data=go.Heatmap(
    z=heatmap_data,
    x=labels,
    y=y_labels,
    text=[[f'{val}%' for val in row] for row in heatmap_data],
    texttemplate='%{text}',
    textfont={"size": 14, "color": "black"},
    colorscale='RdYlBu_r',
    colorbar=dict(title='ユーザーの割合(%)'),
    showscale=True,
))

fig_heatmap.update_layout(
    title='案内表示別スクロール率の分布（最終アクション）',
    xaxis_title='スクロール率範囲',
    yaxis_title='',
    height=250,
)

st.plotly_chart(fig_heatmap, use_container_width=True)

# スクロール率の詳細な統計情報
st.subheader("案内表示別スクロール率の詳細統計（最終アクション）")
col3, col4 = st.columns(2)

with col3:
    st.write("スクロール率の分布")
    for guide in sorted(filtered_df['案内が表示されたか'].unique()):
        st.write(f"案内{guide}")
        guide_df = filtered_df[filtered_df['案内が表示されたか'] == guide]
        guide_df['scroll_range'] = pd.cut(
            guide_df['スクロール率(%)'], bins=bins, labels=labels, include_lowest=True)
        distribution = guide_df['scroll_range'].value_counts().sort_index()
        total = len(guide_df)
        distribution_df = pd.DataFrame({
            'スクロール率範囲': labels,
            'ユーザー数': distribution.values,
            '割合(%)': (distribution / total * 100).round(1).values
        })
        st.dataframe(distribution_df)

with col4:
    st.write("表値")
    stats_data = {}
    for guide in sorted(filtered_df['案内が表示されたか'].unique()):
        guide_df = filtered_df[filtered_df['案内が表示されたか'] == guide]
        stats_data[f'案内{guide}'] = {
            '平均スクロール率': f"{guide_df['スクロール率(%)'].mean():.1f}%",
            '中央値': f"{guide_df['スクロール率(%)'].median():.1f}%",
            '標準偏差': f"{guide_df['スクロール率(%)'].std():.1f}%",
            '最小値': f"{guide_df['スクロール率(%)'].min():.1f}%",
            '最大値': f"{guide_df['スクロール率(%)'].max():.1f}%",
            'ユーザー数': f"{len(guide_df)}人"
        }
    st.write(stats_data)

# デバイスタイプ別の統計
st.subheader("デバイスタイプ別統計")
device_stats = filtered_df.groupby('デバイスタイプ').agg({
    'session ID': 'count',
    'スクロール率(%)': ['mean', 'median', 'std']
}).round(1)

device_stats.columns = ['セッション数', '平均スクロール率', '中央値', '標準偏差']
st.dataframe(device_stats)

# 詳細データの表示
st.subheader("詳細データ（最終アクション）")
st.dataframe(
    filtered_df[[
        'session ID', 'デバイスタイプ', '案内が表示されたか', 'スクロール率(%)',
        'video_time_min', 'セッション時間', 'アクション'
    ]]
)

# LPの構造とスクロール率の可視化
st.subheader("LPの構造とスクロール率の分布")

# カラムレイアウトを調整
col_space, col_scale, col_img, col_analysis = st.columns([2, 0.1, 3.5, 4])

with col_img:
    # LP画像を読み込んでリサイズ
    image = Image.open("pc-lp.png")
    height = 500
    aspect_ratio = image.width / image.height
    width = int(height * aspect_ratio)
    image_resized = image.resize((width, height), Image.LANCZOS)
    st.image(image_resized, use_column_width=False)

with col_scale:
    # スクロール率の目盛りを表示
    fig_scale = go.Figure()

    # 縦線を追加
    fig_scale.add_shape(
        type="line",
        x0=0, x1=0,
        y0=0, y1=100,
        line=dict(color="black", width=2)
    )

    # 目盛りを追加（10%ごと）
    for i in range(0, 101, 10):
        # 横線
        fig_scale.add_shape(
            type="line",
            x0=-0.2, x1=0,
            y0=100-i, y1=100-i,
            line=dict(color="black", width=1)
        )
        # パーセント表示
        fig_scale.add_annotation(
            x=-0.3, y=100-i,
            text=f"{i}%",
            showarrow=False,
            font=dict(size=10)
        )

    fig_scale.update_layout(
        showlegend=False,
        plot_bgcolor='white',
        width=50,
        height=500,
        margin=dict(l=40, r=0, t=0, b=0, pad=0),
        xaxis=dict(
            showgrid=False,
            showticklabels=False,
            range=[-1, 0.2],
            fixedrange=True
        ),
        yaxis=dict(
            showgrid=False,
            showticklabels=False,
            range=[0, 100],
            autorange="reversed",
            fixedrange=True
        )
    )

    st.plotly_chart(fig_scale, use_container_width=True)

with col_analysis:
    # スクロール率の分布分析
    st.write("スクロール率の分布（10%区切り）")

    # スクロール率を10%区切りで集計
    scroll_bins = list(range(0, 101, 10))
    scroll_labels = [f"{i}%-{i+10}%" for i in range(0, 100, 10)]

    # 案内表示の有無でグループ化
    scroll_distribution = pd.DataFrame()
    for guide in sorted(filtered_df['案内が表示されたか'].unique()):
        guide_df = filtered_df[filtered_df['案内が表示されたか'] == guide]
        # スクロール率の範囲を計算
        guide_df['スクロール範囲'] = pd.cut(guide_df['スクロール率(%)'],
                                    bins=scroll_bins,
                                    labels=scroll_labels,
                                    include_lowest=True)
        distribution = guide_df['スクロール範囲'].value_counts().sort_index()
        total = len(guide_df)
        scroll_distribution[f'案内{guide}'] = (distribution / total * 100).round(1)

    # ヒートマップの作成
    fig_scroll_dist = go.Figure(data=go.Heatmap(
        z=scroll_distribution.T.values,
        x=scroll_labels,
        y=[f'案内{guide}' for guide in sorted(filtered_df['案内が表示されたか'].unique())],
        text=[[f'{val}%' for val in row] for row in scroll_distribution.T.values],
        texttemplate='%{text}',
        textfont={"size": 14, "color": "black"},
        colorscale='RdYlBu_r',
        colorbar=dict(title='ユーザーの割合(%)'),
    ))

    fig_scroll_dist.update_layout(
        title='スクロール率の分布（案内表示別）',
        xaxis_title='スクロール率範囲',
        yaxis_title='',
        height=300,
    )

    st.plotly_chart(fig_scroll_dist, use_container_width=True)

    # スクロール範囲別の統計
    st.write("スクロール範囲別の統計")
    stats_df = pd.DataFrame()

    for guide in sorted(filtered_df['案内が表示されたか'].unique()):
        guide_df = filtered_df[filtered_df['案内が表示されたか'] == guide]
        guide_df['スクロール範囲'] = pd.cut(guide_df['スクロール率(%)'],
                                    bins=scroll_bins,
                                    labels=scroll_labels,
                                    include_lowest=True)
        stats_df[f'案内{guide} ユーザー数'] = guide_df['スクロール範囲'].value_counts().sort_index()
        stats_df[f'案内{guide} 割合(%)'] = (guide_df['スクロール範囲'].value_counts().sort_index() / len(guide_df) * 100).round(1)

    st.dataframe(stats_df)
    st.write("※ 各スクロール範囲における最終スクロール位置のユーザー数と割合")
