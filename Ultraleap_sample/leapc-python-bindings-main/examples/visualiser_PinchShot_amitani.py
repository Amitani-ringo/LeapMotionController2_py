import leap
import numpy as np
import cv2

import math
from leap import datatypes as ldt
import json
import os
from datetime import datetime, timedelta

_TRACKING_MODES = {
    leap.TrackingMode.Desktop: "Desktop",
    leap.TrackingMode.HMD: "HMD",
    leap.TrackingMode.ScreenTop: "ScreenTop",
}

def location_end_of_finger(hand: ldt.Hand, digit_idx: int) -> ldt.Vector:
        digit = hand.digits[digit_idx]
        return digit.distal.next_joint

def sub_vectors(v1: ldt.Vector, v2: ldt.Vector) -> list:
    return map(float.__sub__, v1, v2)

def fingers_pinching(thumb: ldt.Vector, index: ldt.Vector):
    diff = list(map(abs, sub_vectors(thumb, index)))

    if diff[0] < 20 and diff[1] < 20 and diff[2] < 20:
        return True, diff
    else:
        return False, diff

def MakeFingerCalibrationFile(globaltime,leaptime,rightthumb: ldt.Vector, rightindex: ldt.Vector):
    # json出力
    fingerdata_toWrite = [
        {
            'global_time': globaltime,
            'leap_time': leaptime,
        },
        {            
            'finger_type': 'right_tumb',
            'position': {
                'x': rightthumb[0],
                'y': rightthumb[1],
                'z': rightthumb[2]
            }
        },
        {
            'finger_type': 'right_index',
            'position': {
                'x': rightindex[0],
                'y': rightindex[1],
                'z': rightindex[2]
            }
        }
    ]
    # JSONファイルのパス
    json_file_path = './CalibrationData/Finger_calibration.json'
    # ディレクトリが存在しなければ作成
    os.makedirs(os.path.dirname(json_file_path), exist_ok=True)
    # JSONファイルへの書き込み
    with open(json_file_path, 'w') as f:
        json.dump(fingerdata_toWrite, f, indent = 2)

def CablibrationShot(fingerdata):
    # 時間関連の処理
    globaltime = datetime.now().strftime('%Y/%m/%d %H:%M:%S.%f')
    leaptime = leap.get_now()
    # 計測データ関連の処理
    # finger data
    Rightthumb = fingerdata[0]
    Rightindex = fingerdata[1]
    MakeFingerCalibrationFile(globaltime,leaptime,Rightthumb, Rightindex)

    # キャリブレーションが終了したことを伝える
    print(f"Making calibrationfiles has been completed!")

class Canvas:
    def __init__(self):
        self.name = "Python Gemini Visualiser+PinchShot_amitani"
        self.screen_size = [500, 700]
        self.hands_colour = (255, 255, 255)
        self.font_colour = (0, 255, 44)
        self.hands_format = "Dots"
        self.output_image = np.zeros((self.screen_size[0], self.screen_size[1], 3), np.uint8)
        self.tracking_mode = None

    def set_tracking_mode(self, tracking_mode):
        self.tracking_mode = tracking_mode

    def toggle_hands_format(self):
        self.hands_format = "Dots" if self.hands_format == "Skeleton" else "Skeleton"
        print(f"Set hands format to {self.hands_format}")

    def get_joint_position(self, bone):
        if bone:
            return int(bone.x + (self.screen_size[1] / 2)), int(bone.z + (self.screen_size[0] / 2))
        else:
            return None

    def put_caption(self, position, level, message):
        font = cv2.FONT_HERSHEY_SIMPLEX
        size = 0.5
        linewidth = 1
        messagesize, baseline = cv2.getTextSize(message,font,size,linewidth)
        if position == "upper_left":
            cv2.putText(
            self.output_image,
            message,
            (10, messagesize[1] * level + 10),
            font,
            size,
            self.font_colour,
            1,
            )
        elif position == "upper_right":
            cv2.putText(
            self.output_image,
            message,
            (self.screen_size[1] - messagesize[0], messagesize[1]*level + 10),
            font,
            size,
            self.font_colour,
            1,
            )
        elif position == "center":
            cv2.putText(
            self.output_image,
            message,
            (math.ceil(self.screen_size[1]/2 - messagesize[0]/2), messagesize[1]*level),
            font,
            size,
            self.font_colour,
            1,
            )

    def PinchShot(self,event):
        # 左手がpinchしているかを監視＋true or falseで返す
        framerate:int = 1
        if event.tracking_frame_id % framerate == 0: # framerateフレームのうち1回
            Flag_leftpinch = False
            Flag_leftcatch = False
            Flag_rightpinch = False
            Flag_rightcatch = False
            for hand in event.hands:
                hand_type = "Left" if str(hand.type) == "HandType.Left" else "Right"
                # 3次元位置
                thumb = location_end_of_finger(hand, 0)
                index = location_end_of_finger(hand, 1)
                # pinching判定
                pinching, array = fingers_pinching(thumb, index) # pinchingしているかどうか

                # 指の距離を円として描画
                # 左手の円の描画
                if hand_type == "Left":
                    # Rightが記録されたことを示すフラグ
                    Flag_leftcatch = True
                    # 示指先端と拇指先端の3次元的な距離
                    length_betweenTumbAndIndex = math.sqrt(array[0] ** 2 + array[1] ** 2 + array[2] ** 3)
                    # 画面上における円の半径
                    r = math.ceil(length_betweenTumbAndIndex * (self.screen_size[0] / (1000)))
                    # 円の中心位置：画面上の拇指地点
                    thumb_position = self.get_joint_position(thumb)
                    index_position = self.get_joint_position(index)
                    circlecenter = np.ceil(np.mean(np.vstack([thumb_position,index_position]),axis=0)).astype(int)
                    # pinch関連のUI
                    if pinching:
                        # pinchした瞬間は塗りつぶし
                        cv2.circle(self.output_image, circlecenter, r, (255, 255, 255),-1)
                        # leftpinchのフラグをtrue
                        Flag_leftpinch = True
                    else:
                        # pinchしていないときは空洞
                        cv2.circle(self.output_image, circlecenter, r, (255, 255, 255))
                    
                # 右手は左手がpinchしているときに強調
                if hand_type == "Right":
                    # Rightが記録されたことを示すフラグ
                    Flag_rightcatch = True
                    # 記録
                    Rightthumb = thumb
                    Rightindex = index
                    Rightthumb_position = self.get_joint_position(thumb)
                    Rightindex_position = self.get_joint_position(index)
            # shot
            # leftpinchingがなされているとき、右手の座標を記録する
            if(not Flag_leftcatch or not Flag_rightcatch):
                # どちらかの手が映っていないとき                
                return False, False
            elif(Flag_leftpinch):
                # 3次元位置のterminalへの出力
                # print(f"Right thumb is located in [{Rightthumb[0]},{Rightthumb[1]},{Rightthumb[2]}]")
                # print(f"Right index is located in [{Rightindex[0]},{Rightindex[1]},{Rightindex[2]}]")
                # 2次元位置の図示
                cv2.circle(self.output_image, Rightthumb_position, 3, (255, 0, 0),-1)
                cv2.circle(self.output_image, Rightindex_position, 3, (255, 0, 0),-1)
                # 3次元位置はcalibration fingerdataとしてまとめる
                calibration_fingerdata = [[Rightthumb[0],Rightthumb[1],Rightthumb[2]],[Rightindex[0],Rightindex[1],Rightindex[2]]]
                
                return True, calibration_fingerdata
            else:
                # 手は映っているが、pinchされていないとき               
                return False, False 
        else: # framerateに該当しない瞬間
            return False,False

    def render_hands(self, event):
        # Clear the previous image
        self.output_image[:, :] = 0

        # 現在のTracking modeを画面左下に緑色で描画
        cv2.putText(
            self.output_image,
            f"Tracking Mode: {_TRACKING_MODES[self.tracking_mode]}",
            (10, self.screen_size[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            self.font_colour,
            1,
        )

        if len(event.hands) == 0:
            return
        
        # 手のレンダリング
        for i in range(0, len(event.hands)): # 見えている手の数だけ、ループする(どうやら手が2本以上認識されることはなさそう)
            hand = event.hands[i]  # hand : 現在見えているi番目の手
            for index_digit in range(0, 5): # 該当する手の0~5ほんの指でループ
                digit = hand.digits[index_digit] # digit : handのinde_digit番目の指
                for index_bone in range(0, 4): # 該当する指の0~4本の骨でループ
                    bone = digit.bones[index_bone] # bone : digitのindex_bone番目の骨
                    # 描画様式：点
                    if self.hands_format == "Dots":
                        prev_joint = self.get_joint_position(bone.prev_joint) # ここにおける関節位置とは、画面上における位置(3Dではない)
                        next_joint = self.get_joint_position(bone.next_joint)
                        if prev_joint:
                            # 関節の位置
                            cv2.circle(self.output_image, prev_joint, 2, self.hands_colour, -1)

                        if next_joint:
                            # 指先の位置
                            cv2.circle(self.output_image, next_joint, 2, (0, 255, 255), -1)
                    #　描画様式：点＋線
                    if self.hands_format == "Skeleton":
                        wrist = self.get_joint_position(hand.arm.next_joint)
                        elbow = self.get_joint_position(hand.arm.prev_joint)
                        if wrist:
                            cv2.circle(self.output_image, wrist, 3, self.hands_colour, -1)

                        if elbow:
                            cv2.circle(self.output_image, elbow, 3, self.hands_colour, -1)

                        if wrist and elbow:
                            cv2.line(self.output_image, wrist, elbow, self.hands_colour, 2)

                        bone_start = self.get_joint_position(bone.prev_joint)
                        bone_end = self.get_joint_position(bone.next_joint)

                        if bone_start:
                            cv2.circle(self.output_image, bone_start, 3, self.hands_colour, -1)

                        if bone_end:
                            cv2.circle(self.output_image, bone_end, 3, self.hands_colour, -1)

                        if bone_start and bone_end:
                            cv2.line(self.output_image, bone_start, bone_end, self.hands_colour, 2)

                        if ((index_digit == 0) and (index_bone == 0)) or (
                            (index_digit > 0) and (index_digit < 4) and (index_bone < 2)
                        ):
                            index_digit_next = index_digit + 1
                            digit_next = hand.digits[index_digit_next]
                            bone_next = digit_next.bones[index_bone]
                            bone_next_start = self.get_joint_position(bone_next.prev_joint)
                            if bone_start and bone_next_start:
                                cv2.line(
                                    self.output_image,
                                    bone_start,
                                    bone_next_start,
                                    self.hands_colour,
                                    2,
                                )

                        if index_bone == 0 and bone_start and wrist:
                            cv2.line(self.output_image, bone_start, wrist, self.hands_colour, 2)


class TrackingListener(leap.Listener):
    def __init__(self, canvas):
        self.canvas = canvas

    def on_connection_event(self, event):
        pass

    def on_tracking_mode_event(self, event):
        self.canvas.set_tracking_mode(event.current_tracking_mode)
        print(f"Tracking mode changed to {_TRACKING_MODES[event.current_tracking_mode]}")

    def on_device_event(self, event):
        try:
            with event.device.open():
                info = event.device.get_info()
        except leap.LeapCannotOpenDeviceError:
            info = event.device.get_info()

        print(f"Found device {info.serial}")

    def on_tracking_event(self, event):
        global Calibration_ready, Calibration_starttime, Calibration_done
        self.canvas.render_hands(event) # trackingが成立しているときにrender_handsを実行
        Finger_shot = False
        calibrationfingerdata = False
        Finger_shot, calibrationfingerdata = self.canvas.PinchShot(event) # Pinch動作でshot
        
        # message関連
        if Finger_shot and not Calibration_done:
            self.canvas.put_caption("upper_right", 1, "Please wait...")
        elif not Finger_shot and not Calibration_done:
            self.canvas.put_caption("upper_right", 1, "Move both hand into frame and pinch your left hand.")
        elif Calibration_done:
            self.canvas.put_caption("upper_right", 1, "Calibration has been completed.")
            self.canvas.put_caption("upper_right", 2, "Press x key to close this window.")
            

        # Shot関連の処理
        if Finger_shot and not Calibration_ready:# 初めてpinchされたとき            
            Calibration_ready = True
            Calibration_starttime = datetime.now() + timedelta(seconds=3)
            print(f"Pinching has been observed. Calibration will be conducted in 3s.")
        elif not Finger_shot and Calibration_ready:
            Calibration_ready = False
        
        # calibrationを開始する
        if Calibration_ready and datetime.now() > Calibration_starttime and not Calibration_done:
            CablibrationShot(calibrationfingerdata)
            print(f"Calibration shot has completed.")
            Calibration_done = True
            Calibration_ready = False
        elif Calibration_ready and not datetime.now() > Calibration_starttime and not Calibration_done:
            Tdiff = Calibration_starttime - datetime.now() + timedelta(seconds=1)
            self.canvas.put_caption("center", 9, f"Pinching has been observed. Calibration will be conducted in 3s.")
            self.canvas.put_caption("center", 10, f"{Tdiff.seconds}")
        elif Calibration_done:
            self.canvas.put_caption("upper_right", 39, f"Calibration shot has completed.")
            self.canvas.put_caption("upper_right", 40, f"Filename : Finger_calibration.json")
            

def main():
    # キャリブレーション実行関連のグローバル変数
    global Calibration_ready
    Calibration_ready = False
    global Calibration_starttime
    Calibration_starttime = datetime.now()
    global Calibration_done
    Calibration_done = False

    canvas = Canvas()

    # terminalに出力される注意書き
    print(canvas.name)
    print("")
    print("Press <key> in visualiser window to:")
    print("  x: Exit")
    print("  h: Select HMD tracking mode")
    print("  s: Select ScreenTop tracking mode")
    print("  d: Select Desktop tracking mode")
    print("  f: Toggle hands format between Skeleton/Dots")
    
    tracking_listener = TrackingListener(canvas)

    connection = leap.Connection()
    connection.add_listener(tracking_listener)

    running = True

    with connection.open():
        connection.set_tracking_mode(leap.TrackingMode.Desktop)
        canvas.set_tracking_mode(leap.TrackingMode.Desktop)

        while running:
            cv2.imshow(canvas.name, canvas.output_image) # on_tracking_eventで事前に設定されたcanvasの様子を描画

            key = cv2.waitKey(1)

            if key == ord("x"): # 終了
                break
            elif key == ord("h"): # Trackingm mode をHMDにする
                connection.set_tracking_mode(leap.TrackingMode.HMD)
            elif key == ord("s"): # Tracking mode をScreen topにする
                connection.set_tracking_mode(leap.TrackingMode.ScreenTop)
            elif key == ord("d"): # Tracking mode をDesktopにする
                connection.set_tracking_mode(leap.TrackingMode.Desktop)
            elif key == ord("f"): # 表示形式をドット＋線にするかドットのみにするかを切り替える
                canvas.toggle_hands_format()


if __name__ == "__main__":
    main()
