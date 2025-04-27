from ultralytics import YOLO
import numpy as np
from metrics_functions import *
import statistics as stats
from operator import itemgetter

if __name__ == '__main__':

    # Perform object detection on an image
    model=YOLO('runs\segment/train12\weights/best.pt')
    results = model("demonstration")  # Predict on an image
    results[0].show()  # Display results
    results[1].show()

    idx=0

    for result in results:
        #print("result.tojson:", result.tojson())
        #print(result)
        idx+=1
        print(f'Image {idx}:')

        name=result[0].names #{0: 'knot', 1: 'leftover', 2: 'stitch', 3: 'wound'}
        #print(name)

        names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  # class name of each detected object

        #box=result[0]
        bxy = result.boxes.xyxy
        bxyn = result.boxes.xyxyn #normalised
        bxywh = result.boxes.xywh #width and height
        bxywhn = result.boxes.xywhn #width and height normalised

        xy = result.masks.xy  # mask in polygon format
        xyn = result.masks.xyn  # normalized
        masks = result.masks.data  # mask in matrix format (num_objects x H x W)

        #### KNOT PLACEMENT

        knots=[]
        knot_threshold=0.01
        leftovers=[]
        stitches=[]
        wounds=[]
        min_angle_thres=80
        max_angle_thres=100
        skew_thres=15
        length_thres=0.05
        dist_thres=0.05
        lolen_thres=0.025
        bitesl_thres=0.03
        bitesr_thres=0.03
        bites_thres=0.03

        for i in range(len(names)):
            #print(f'{names[i]}: {xyn[i]}')

            if names[i]=='knot':
                knots_aux=[]
                for j in range(len(xyn[i])):
                    knots_aux.append(xyn[i][j][0])  #EXTRACT THE AVERAGE OF X
                knots.append(avg(knots_aux))

            if names[i]=='leftover':
                leftovers_aux=[[2.0,0.0],[-1.0,0.0]]  #any x stitch will be less than 2 and more than -1
                for j in range(len(xyn[i])):
                    if xyn[i][j][0]<leftovers_aux[0][0]:  #EXTRACT MIN X
                        leftovers_aux[0]=xyn[i][j]
                    if xyn[i][j][0]>leftovers_aux[1][0]:  #EXTRACT MAX X
                        leftovers_aux[1]=xyn[i][j]   
                leftovers+=leftovers_aux   #EXTRACT MAX AND MIN X

            if names[i]=='stitch':
                stitches_aux=[[2.0,0.0],[-1.0,0.0]]  #any x stitch will be less than 2 and more than -1
                for j in range(len(xyn[i])):
                    if xyn[i][j][0]<stitches_aux[0][0]:  #EXTRACT MIN X
                        stitches_aux[0]=xyn[i][j]
                    if xyn[i][j][0]>stitches_aux[1][0]:  #EXTRACT MAX X
                        stitches_aux[1]=xyn[i][j]   
                stitches+=stitches_aux   #EXTRACT MAX AND MIN X

            if names[i]=='wound':
                wounds_aux=[[0,2],[0,-1]]  #any x stitch will be less than 2 and more than -1
                for j in range(len(xyn[i])):
                    if xyn[i][j][1]<wounds_aux[0][1]:  #EXTRACT MIN Y
                        wounds_aux[0]=xyn[i][j]
                    if xyn[i][j][1]>wounds_aux[1][1]:  #EXTRACT MAX Y
                        wounds_aux[1]=xyn[i][j]   
                wounds+=wounds_aux   #EXTRACT MAX AND MIN Y

        sorted_stitches=sorted(stitches[:], key=itemgetter(1))
        for st in range(0,len(sorted_stitches),2):
            a=sorted_stitches[st]
            if sorted_stitches[st][0]>sorted_stitches[st+1][0]:
                sorted_stitches[st]=sorted_stitches[st+1]
                sorted_stitches[st+1]=a

        #### KNOT PLACEMENT

        if knots==[]:
            print('Error: No knots detected.')
        else:
            avg_knots=avg(knots)
            knot_success=0
            for knot in knots:
                if abs(knot-avg_knots)<knot_threshold:
                    knot_success+=1
            knot_quality=(knot_success/len(knots))*100
            print(f'Knot Placement Quality: {knot_quality:.2f}%')
        
        #### SUTURE ORIENTATION (PERPENDICULARITY OF STITCHES AND WOUND)

        if wounds==[]:
            print('Error: No wound detected.')
        elif sorted_stitches==[]:
            print('Error: No stitches detected.')
        else:
            w=wound_orientation(wounds[0][0],wounds[0][1],wounds[1][0],wounds[1][1])
            angle_success=0

            for st in range(0,len(sorted_stitches),2):
                s=stitch_orientation(sorted_stitches[st][0],sorted_stitches[st][1],sorted_stitches[st+1][0],sorted_stitches[st+1][1])
                angle=abs(w-s)
                if angle>min_angle_thres and angle<max_angle_thres:
                    angle_success+=1
            angle_quality=(angle_success/len(sorted_stitches)*2)*100
            print(f'Suture Orientation Quality (Perpendicularity Between the Stitch and the Wound): {angle_quality:.2f}%')


        #### SKEWNESS OF ANGLE (UNIFORMITY BETWEEN STITCHES' ANGLES)

        if sorted_stitches==[]:
            print('Error: No stitches detected.')
        else:
            skew_success=0
            for st in range(0,len(sorted_stitches)-2,2):
                s1=stitch_orientation(sorted_stitches[st][0],sorted_stitches[st][1],sorted_stitches[st+1][0],sorted_stitches[st+1][1])
                s2=stitch_orientation(sorted_stitches[st+2][0],sorted_stitches[st+2][1],sorted_stitches[st+3][0],sorted_stitches[st+3][1])

                skew=abs(s1-s2)
                if skew<skew_thres:
                    skew_success+=1
            
            skew_quality=(skew_success/(len(sorted_stitches)-1)*2)*100
            print(f"Skewness of Angle Quality (Stitch Orientation): {skew_quality:.2f}%")

        #### STITCH BITE UNIFORMITY

        lengths=[]
        length_success=0

        if sorted_stitches==[]:
            print('Error: No stitches detected.')
        else:
            for st in range(0,len(sorted_stitches),2):
                l=length(sorted_stitches[st][0],sorted_stitches[st][1],sorted_stitches[st+1][0],sorted_stitches[st+1][1])
                lengths.append(l)

            avg_length=avg(lengths)
            for i in range(len(lengths)):
                if length_thres>abs(lengths[i]-avg_length):
                    length_success+=1
            length_quality=(length_success/len(lengths))*100
            print(f'Bite Size Quality (Stitch Size): {length_quality:.2f}%')

        #### PITCH (STITCH DISTANCE) UNIFORMITY

        distances=[]
        distance_success=0

        if sorted_stitches==[]:
            print('Error: No stitches detected.')
        else:
            for st in range(0,len(sorted_stitches)-2,2):
                d=[0,0]
                d1=length(sorted_stitches[st][0],sorted_stitches[st][1],sorted_stitches[st+2][0],sorted_stitches[st+2][1])
                d2=length(sorted_stitches[st+1][0],sorted_stitches[st+1][1],sorted_stitches[st+3][0],sorted_stitches[st+3][1])
                d[0]=d1
                d[1]=d2
                avg_d=avg(d)
                distances.append(avg_d)

            avg_distance=avg(distances)
            for i in range(len(distances)):
                if dist_thres>abs(distances[i]-avg_distance):
                    distance_success+=1
            distance_quality=(distance_success/len(distances))*100
            print(f'Pitch Size Quality (Distance Between Stitches): {distance_quality:.2f}%')

        #### LEFTOVER LENGTH UNIFORMITY

        lo_lengths=[]
        lolen_success=0

        if leftovers==[]:
            print('Error: No leftovers detected.')
        else:
            for lo in range(0,len(leftovers),2):
                l=length(leftovers[lo][0],leftovers[lo][1],leftovers[lo+1][0],leftovers[lo+1][1])
                lo_lengths.append(l)

            avg_lolen=avg(lo_lengths)
            for i in range(len(lo_lengths)):
                if lolen_thres>abs(lo_lengths[i]-avg_lolen):
                    lolen_success+=1
            lolen_quality=(lolen_success/len(lo_lengths))*100
            print(f'Leftover Size Quality: {lolen_quality:.2f}%')

        #### BITE SIZE ON LEFT SIDE

        bitesl=[]
        bitesl_success=0

        if sorted_stitches==[]:
            print('Error: No stitches detected.')
        else:
            for st in range(0,len(sorted_stitches)-2,2):
                bl=distance(sorted_stitches[st][0],sorted_stitches[st+2][0])
                bitesl.append(bl)

            avg_bitesl=avg(bitesl)
            for i in range(len(bitesl)):
                if bitesl_thres>abs(bitesl[i]-avg_bitesl):
                    bitesl_success+=1
            bitesl_quality=(bitesl_success/len(bitesl))*100
            print(f'Left Bite Size Quality: {bitesl_quality:.2f}%')

        #### BITE SIZE ON RIGHT SIDE

        bitesr=[]
        bitesr_success=0

        if sorted_stitches==[]:
            print('Error: No stitches detected.')
        else:
            for st in range(0,len(sorted_stitches)-2,2):
                br=distance(sorted_stitches[st+1][0],sorted_stitches[st+3][0])
                bitesr.append(br)

            avg_bitesr=avg(bitesr)
            for i in range(len(bitesr)):
                if bitesr_thres>abs(bitesr[i]-avg_bitesr):
                    bitesr_success+=1
            bitesr_quality=(bitesr_success/len(bitesr))*100
            print(f'Right Bite Size Quality: {bitesr_quality:.2f}%')

        #### OVERALL BITE SIZE = OVERALL SYMMETRY

        if sorted_stitches==[]:
            print('Error: No stitches detected.')
        else:
            bites_success=bitesr_success+bitesl_success
            bites_length=len(bitesl)+len(bitesr)
            bites_quality=(bites_success/bites_length)*100
            print(f'Overall Suture Symmetry: {bites_quality:.2f}%')

        #### OVERALL QUALITY OF SUTURE

        overall_quality_len=8

        if knots==[]:
            knot_quality=0
            overall_quality_len-=1
        elif wounds==[]:
            angle_quality=0
            overall_quality_len-=1
            
        suture_quality=(knot_quality+angle_quality+lolen_quality+bitesl_quality+bitesr_quality+length_quality+distance_quality+skew_quality)/overall_quality_len
        if suture_quality<50:
            grade='Inadequate'
        elif 50<suture_quality<70:
            grade='Adequate'
        elif 70<suture_quality<80:
            grade='Good'
        elif 80<suture_quality<90:
            grade='Very Good'
        else:
            grade='Excelent'
        print(f'Overall Suture Quality: {grade} ({suture_quality:.2f}%)')

        print('----------------------------------------------')