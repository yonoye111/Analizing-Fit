import sys
import os
import fitparse
import numpy as np
import pandas as pd
import folium
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import radians, cos, sin, asin, sqrt

def haversine(lon1, lat1, lon2, lat2):
    """
    2点間の距離を計算(メートル単位)
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371000  # 地球の半径(メートル)
    return c * r

def semicircle_to_degrees(semicircles):
    """
    FITファイルのセミシークル座標を度に変換する
    """
    if semicircles is None:
        return None
    return semicircles * (180.0 / 2**31)

def parse_fit_file(fit_file_path):
    """
    FITファイルを読み込み、Pandas DataFrameとしてデータを抽出する
    """
    fitfile = fitparse.FitFile(fit_file_path)
    
    data = []
    for record in fitfile.get_messages('record'):
        row = {}
        for data_point in record:
            row[data_point.name] = data_point.value
        data.append(row)
        
    df = pd.DataFrame(data)
    return df

def create_map(df, output_html='map.html'):
    """
    FITファイルの座標データからOpenStreetMap上にルートを描画する
    """
    # 緯度・経度の列名を確認（position_lat / position_long）
    lat_col = 'position_lat' if 'position_lat' in df.columns else None
    lon_col = 'position_long' if 'position_long' in df.columns else None

    if not lat_col or not lon_col:
        print("エラー: FITファイルに位置情報（緯度・経度）が見つかりませんでした")
        return

    # 有効な座標データを抽出して度に変換
    valid_df = df.dropna(subset=[lat_col, lon_col]).copy()
    valid_df['lat'] = valid_df[lat_col].apply(semicircle_to_degrees)
    valid_df['lon'] = valid_df[lon_col].apply(semicircle_to_degrees)

    coordinates = valid_df[['lat', 'lon']].values.tolist()

    if not coordinates:
        print("エラー: 有効な座標データが見つかりませんでした")
        return

    center_lat = sum(c[0] for c in coordinates) / len(coordinates)
    center_lon = sum(c[1] for c in coordinates) / len(coordinates)

    # Foliumマップの作成（国土地理院タイルを使用）
    map_object = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles='https://cyberjapandata.gsi.go.jp/xyz/std/{z}/{x}/{y}.png',
        attr='国土地理院',
        dragging=True,
        scroll_wheel_zoom=True,
        zoom_control=True
    )

    # ルートを赤線で描画
    folium.PolyLine(
        coordinates,
        color='red',
        weight=3,
        opacity=0.8
    ).add_to(map_object)

    # スタート・ゴールマーカー
    if coordinates:
        folium.Marker(coordinates[0], popup='スタート', icon=folium.Icon(color='green', icon='play')).add_to(map_object)
        if len(coordinates) > 1:
            folium.Marker(coordinates[-1], popup='ゴール', icon=folium.Icon(color='red', icon='stop')).add_to(map_object)

    map_object.save(output_html)
    print(f"地図を {output_html} に保存しました (ポイント数: {len(coordinates)})")

def create_interactive_plots(df, output_html='data_plot.html'):
    """
    FITファイルから速度、心拍数、標高、ケイデンスのインタラクティブなグラフを作成する
    """
    # タイムスタンプや各データの取得
    time_col = 'timestamp' if 'timestamp' in df.columns else None
    ele_col = 'enhanced_altitude' if 'enhanced_altitude' in df.columns else ('altitude' if 'altitude' in df.columns else None)
    hr_col = 'heart_rate' if 'heart_rate' in df.columns else None
    cad_col = 'cadence' if 'cadence' in df.columns else None
    speed_col = 'enhanced_speed' if 'enhanced_speed' in df.columns else ('speed' if 'speed' in df.columns else None)

    # 欠損値補完やデフォルト配列の準備
    n_rows = len(df)
    
    # 時間軸（秒経過）の作成
    if time_col and not df[time_col].isnull().all():
        times = pd.to_datetime(df[time_col])
        time_seconds = (times - times.iloc[0]).dt.total_seconds().values
    else:
        time_seconds = np.arange(n_rows)

    # 速度 (m/s -> km/h)
    if speed_col and not df[speed_col].isnull().all():
        speeds_kmh = df[speed_col].fillna(0).values * 3.6
    else:
        speeds_kmh = np.zeros(n_rows)

    # 心拍数
    hr_values = df[hr_col].fillna(0).values if hr_col and hr_col in df.columns else np.zeros(n_rows)

    # 標高
    ele_values = df[ele_col].fillna(0).values if ele_col and ele_col in df.columns else np.zeros(n_rows)

    # ケイデンス
    cadence_values = df[cad_col].fillna(0).values if cad_col and cad_col in df.columns else np.zeros(n_rows)

    # 4段のサブプロット作成
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=('速度 (Velocity)', '心拍数 (Heart Rate)', '標高 (Elevation)', 'ケイデンス (Cadence)'),
        vertical_spacing=0.08,
        shared_xaxes=True
    )

    # 1段目: 速度
    fig.add_trace(go.Scatter(x=time_seconds, y=speeds_kmh, mode='lines', name='速度', line=dict(color='#2563eb', width=2)), row=1, col=1)
    # 2段目: 心拍数
    fig.add_trace(go.Scatter(x=time_seconds, y=hr_values, mode='lines', name='心拍数', line=dict(color='#dc2626', width=2)), row=2, col=1)
    # 3段目: 標高
    fig.add_trace(go.Scatter(x=time_seconds, y=ele_values, mode='lines', name='標高', line=dict(color='#16a34a', width=2)), row=3, col=1)
    # 4段目: ケイデンス
    fig.add_trace(go.Scatter(x=time_seconds, y=cadence_values, mode='lines', name='ケイデンス', line=dict(color='#ca8a04', width=2)), row=4, col=1)

    fig.update_xaxes(title_text="時間 [s]", row=4, col=1)
    fig.update_yaxes(title_text="速度 [km/h]", row=1, col=1)
    fig.update_yaxes(title_text="心拍数 [bpm]", row=2, col=1)
    fig.update_yaxes(title_text="標高 [m]", row=3, col=1)
    fig.update_yaxes(title_text="ケイデンス [rpm]", row=4, col=1)

    fig.update_layout(
        autosize=True,
        margin=dict(l=50, r=20, t=50, b=40),
        showlegend=False,
        title_text="FITデータ分析",
        hovermode='x unified',
        dragmode='pan'
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')

    # 統計情報の表示
    print(f"\n速度の統計情報:")
    if len(speeds_kmh) > 0:
        print(f"最大速度: {np.max(speeds_kmh):.2f} km/h")
        print(f"平均速度: {np.mean(speeds_kmh):.2f} km/h")

    if len(ele_values) > 0 and np.any(ele_values > 0):
        print(f"\n標高の統計情報:")
        print(f"最大標高: {np.max(ele_values):.1f} m")
        print(f"最小標高: {np.min(ele_values):.1f} m")

    if len(cadence_values) > 0:
        cad_nonzero = cadence_values[cadence_values > 0]
        if len(cad_nonzero) > 0:
            print(f"\nケイデンスの統計情報:")
            print(f"平均ケイデンス: {np.mean(cad_nonzero):.1f} rpm")
            print(f"最大ケイデンス: {np.max(cad_nonzero):.1f} rpm")

    fig.write_html(
        output_html,
        config={'scrollZoom': True, 'responsive': True},
        default_height='100%',
        default_width='100%'
    )
    print(f"インタラクティブなプロットを '{output_html}' に保存しました")
    return fig

def create_combined_view(fit_file_path, output_html='fit_analysis.html'):
    """
    地図とグラフを1つのHTMLに統合するビューを作成する
    """
    df = parse_fit_file(fit_file_path)
    
    output_dir = os.path.dirname(output_html) if os.path.dirname(output_html) else '.'
    base_name = os.path.splitext(os.path.basename(output_html))[0]

    map_temp = os.path.join(output_dir, f'{base_name}_map.html')
    plot_temp = os.path.join(output_dir, f'{base_name}_plot.html')

    create_map(df, map_temp)
    create_interactive_plots(df, plot_temp)

    combined_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FITデータ分析 - 地図とグラフ</title>
    <style>
        @page {{ size: A4 landscape; margin: 0; }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            background-color: #e5e7eb;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
        }}
        .container {{
            display: flex;
            flex-direction: column;
            width: 297mm;
            height: 210mm;
            background-color: white;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
            color: white;
            padding: 8px 15px;
            flex-shrink: 0;
        }}
        .header h1 {{ margin: 0; font-size: 18px; font-weight: 600; }}
        .info {{
            padding: 5px 15px;
            background-color: #e0e7ff;
            border-bottom: 1px solid #c7d2fe;
            font-size: 11px;
            color: #4338ca;
            flex-shrink: 0;
        }}
        .content {{ display: flex; flex: 1; overflow: hidden; min-height: 0; }}
        .map-container {{ flex: 1; position: relative; border-right: 2px solid #ddd; }}
        .plot-container {{ flex: 1.2; position: relative; background-color: white; }}
        iframe {{ width: 100%; height: 100%; border: none; display: block; }}
        .section-title {{
            position: absolute;
            top: 8px;
            left: 8px;
            background-color: rgba(255, 255, 255, 0.9);
            padding: 4px 10px;
            border-radius: 4px;
            font-weight: 600;
            font-size: 12px;
            color: #333;
            box-shadow: 0 1px 4px rgba(0,0,0,0.15);
            z-index: 1000;
            pointer-events: none;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚴 FITデータ分析</h1>
        </div>
        <div class="info">
            <strong>📍 操作方法:</strong>
            地図: ドラッグで移動、ホイールでズーム | グラフ: ドラッグでパン、ホイールでズーム、ダブルクリックでリセット
        </div>
        <div class="content">
            <div class="map-container">
                <div class="section-title">📍 ルート地図</div>
                <iframe src="{os.path.basename(map_temp)}"></iframe>
            </div>
            <div class="plot-container">
                <div class="section-title">📊 データグラフ</div>
                <iframe src="{os.path.basename(plot_temp)}"></iframe>
            </div>
        </div>
    </div>
</body>
</html>"""

    with open(output_html, 'w', encoding='utf-8') as f:
        f.write(combined_html)

    print(f"\n✅ 統合ビューを '{output_html}' に保存しました")

def main():
    if len(sys.argv) < 2:
        print("使用方法: python fit_analyzer.py <fitファイルパス> [出力HTMLファイル名]")
        print("例: python fit_analyzer.py activity.fit")
        sys.exit(1)

    fit_path = sys.argv[1]
    output_html = sys.argv[2] if len(sys.argv) > 2 else 'fit_analysis.html'

    print(f"FITファイルを解析中: {fit_path}")
    print("=" * 60)
    create_combined_view(fit_path, output_html)
    print("\n" + "=" * 60)
    print("解析完了！ブラウザで出力ファイルを確認してください。")

if __name__ == '__main__':
    main()