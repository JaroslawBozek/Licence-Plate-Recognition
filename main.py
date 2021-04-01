import argparse
import json
from pathlib import Path
import cv2
import numpy as np



#(Unused) This function was used to prepare letters which program compares results with.
def cropping50x50():
    img1 = cv2.imread('input_letter.PNG', 0)
    img1 = cv2.bitwise_not(img1)
    ret, img1 = cv2.threshold(img1, 0, 255, cv2.THRESH_BINARY)

    result = np.where(img1 == 255)
    print(result)
    left = min(result[1])
    right = max(result[1])
    top = min(result[0])
    bottom = max(result[0])
    img1 = img1[top:bottom, left:right]
    img1 = cv2.resize(img1, (50, 50))
    ret, img1 = cv2.threshold(img1, 0, 255, cv2.THRESH_BINARY)

    cv2.imwrite('output_letter.PNG', img1)
    cv2.waitKey()

#(Unused) This function was used to analyze templates and their bottom/top half pixels ratio
def letter_test():
    letter_images = []
    letters_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                    'K', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    for i in range(len(letters_list)):
        img_name = letters_list[i] + '_crop.PNG'
        img = cv2.imread(img_name, 0)
        letter_images.append(img)

        sum = np.sum(img == 255)
        print(letters_list[i], i, sum)
        bottom_sum = np.sum(img[25:,:] == 255)
        top_sum = np.sum(img[:25,:] == 255)
        print(letters_list[i], i, bottom_sum/top_sum)


#Main function
def FindSomeLetters(image):
    #Calculation time may vary. Heavily depends on number of details

    #1 - Load all pattern images and add them to array
    letter_images = []
    letters_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                    'K', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    for i in range(len(letters_list)):
        img_name = letters_list[i] + '_crop.PNG'
        img = cv2.imread('Patterns/' + img_name, 0)
        letter_images.append(img)


    #2 - Load an image and resize it
    x,y,c = image.shape
    mult = 1

    if y > 1000:
        mult = 1
    if y > 2000:
        mult = 0.5
    if y > 3000:
        mult = 0.33
    if y > 4000:
        mult = 0.25
    image = cv2.resize(image, None, fx=mult, fy=mult)

    image_test = image.copy()
    image_test_cp = image.copy()


    last_loop_contours = []
    hall_of_fame = []
    hof_score = []

    diff_tolerance = image.shape[1]/250 #indicates the tolerance of changes in bounding boxes dimensions over iterations. Depends on letter sizes
    iterations = 85 #the more, the better (cap 200)
    test_mode = 0 #change to 1 to analyze the results
    possible_plates_enabled = 1
    #3 - Perform different threshoildings 'iterations' times and count how many times did certain bounding boxes reappear
    for div in range(iterations):
        if test_mode == 1:
            image_test[:] = image_test_cp[:]

        thresh = 50+div*(200/iterations)
        #image gets thresholded on every iteration with a different value (ranges from 50 to 250)
        image_threshold = cv2.inRange(image, (0, 50, thresh), (255, 255, 255))

        kernel = np.ones((3, 3), np.uint8)
        image_threshold = cv2.morphologyEx(image_threshold, cv2.MORPH_OPEN, kernel, iterations=1)
        image_threshold = cv2.filter2D(image_threshold, -1, kernel)
        image_threshold = cv2.bitwise_not(image_threshold)

        #find contours

        contours, hierarchy = cv2.findContours(image_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if test_mode == 1:
            cv2.drawContours(image_test, contours, -1, (0, 255, 0), 3)

        #necessery to prevent program from crashing
        if len(contours) != 0:
            hier = hierarchy[0][:,3]
        else:
            hier = []

        considered_contours = []

        #analyze all contours
        for i in range(len(contours)):
            #create bounding rectangles
            x, y, w, h = cv2.boundingRect(contours[i])

            #consider contours with letter-like dimensions
            #it's a really easy condition to pass, it eliminates things like long plates, and really small objects
            #it depends on plate/image ratio so it won't work on weirdly cropped images
            #h/image.shape[1] > 0.05 condition is algorithm's weakness because it really depends on it
            if h/image.shape[1] > 0.05 and w/image.shape[0] < 0.25 and w/h < 1.5:
                x_par, y_par, w_par, h_par = cv2.boundingRect(contours[hier[i]])

                deny = 0
                #remove doubled boxes
                if h_par/h < 5.0 and h_par/h > 1.0 and w_par/w < 2.0 and w_par/w > 1.0:
                    deny = 1
                #remove boxes colliding with image edges
                if x <= 3 or x + w + 3 >= image.shape[1]:
                    deny = 1

                #if contour was accepted, add it to the list
                if deny == 0:
                    if test_mode == 1:
                        cv2.rectangle(image_test, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    considered_contours.append([x,y,w,h,thresh])

        cuts = 50
        unique_contours = []

        for i in range(cuts-1):
            cut = ((i+1)/cuts)*image.shape[0]
            contours_in_cut = 0
            potential_contours = []
            for j in considered_contours:
                if (cut > j[1] and cut < j[3] + j[1]):
                    contours_in_cut = contours_in_cut + 1
                    potential_contours.append(j)

            if test_mode == 1:
                image_test[int(cut),:] = [255, 0, 0]
            if contours_in_cut >= 4:
                for j in potential_contours:
                    if j not in unique_contours:
                        unique_contours.append(j)

        for num_i, i in enumerate(unique_contours):
            for num_j, j in enumerate(last_loop_contours):
                if abs(i[0] - j[0]) <= diff_tolerance and abs(i[1] - j[1]) <= diff_tolerance and abs(i[2] - j[2]) <= diff_tolerance and abs(i[3] - j[3]) <= diff_tolerance:


                    where = -1
                    for num_k, k in enumerate(hall_of_fame):
                        if j == k:
                            where = num_k

                    if where == -1:
                        hall_of_fame.append(i)
                        hof_score.append(1)
                    else:
                        hof_score[where] = hof_score[where] + 1
                        hall_of_fame[where] = i

        if test_mode == 1:
            cv2.imshow('image_test', image_test)
            cv2.waitKey()
        last_loop_contours = unique_contours


    if len(hof_score) == 0 and len(hall_of_fame) == 0:
        return '???????'

    #PART 4 - find <= 7 bounding boxes with the best scores and count them as letters
    hof_score, hall_of_fame = zip(*sorted(zip(hof_score, hall_of_fame)))


    hall_of_fame = hall_of_fame[-7:]
    left_sides = []
    for i in hall_of_fame:
        left_sides.append(i[0])

    left_sides, hall_of_fame = zip(*sorted(zip(left_sides, hall_of_fame)))
    letters7 = []

    #5 - Perform hsv and otsu thresholding on letters
    for iteration, i in enumerate(hall_of_fame):
        name = str(iteration)


        letter = image[i[1]:i[1]+i[3],i[0]:i[0]+i[2]]
        hsv_letter = cv2.cvtColor(letter, cv2.COLOR_BGR2HSV)
        hsv_letter = cv2.inRange(hsv_letter, (0, 0, i[4]), (255, 255, 255))
        hsv_letter = cv2.bitwise_not(hsv_letter)

        otsu_letter = cv2.cvtColor(letter, cv2.COLOR_BGR2GRAY)

        # otsu binarization
        blur = cv2.GaussianBlur(otsu_letter, (3, 3), 0)
        ret3, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        otsu_letter = cv2.bitwise_not(otsu)

        #the output is a xor of otsu binarization and last hsv binarization
        letter = cv2.bitwise_and(hsv_letter, otsu_letter)
        letters7.append(letter)
        if test_mode == 1:
            # cv2.imshow(name+'otsu', otsu_letter)
            # cv2.imshow(name+'hsv', hsv_letter)
            cv2.imshow(name, letter)


    # get letters
    output_string = ''
    sum_lists = []
    real_chain = ''
    chain = ''

    #6 - Perform watershed, warp perspective and try to compare letters to patters using xor and some extra pixel
    #comparisons. The letters with the patterns score are chosen.
    for i in range(len(letters7)):

        #get rid of noise
        letter_copy = letters7[i].copy()

        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(letters7[i], cv2.MORPH_OPEN, kernel, iterations=1)
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform, 0.05 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        ret, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        labels = np.unique(markers)[2:]

        #find biggest label
        size = 0
        biggest_label = -1
        for index, label in enumerate(labels):

            size_label = len(letter_copy[markers == label])
            if size_label > size:
                size = size_label
                biggest_label = index

        letter_copy[:] = 0
        letter_copy[markers == biggest_label+2] = 255


        contours_letter, hierarchy_letter = cv2.findContours(letter_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours_letter) == 0:
            continue
        rect = cv2.minAreaRect(contours_letter[0])
        box = cv2.boxPoints(rect)
        box = np.int0(box)


        width = int(rect[1][0])
        height = int(rect[1][1])

        src_pts = box.astype("float32")
        dst_pts = np.array([[0, height - 1],
                            [0, 0],
                            [width - 1, 0],
                            [width - 1, height - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        warped = cv2.warpPerspective(letter_copy, M, (width, height))
        letter_h, letter_w = warped.shape
        if letter_w > letter_h:
            warped = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)
        letters7[i] = warped


        tested = cv2.resize(letters7[i], (50, 50))

        sum_list = []
        for j in range(len(letters_list)):
            out_xor = cv2.bitwise_xor(tested, letter_images[j])
            letter_sum = np.sum(out_xor == 255)
            sum_list.append(letter_sum)


        # The ratios were calculated in the unused letter_test() function.
        # If letter's ratio is located in specified range, all possible letters get bonus to their score.
        # Obviously they won't have the same ratio as originals, so the range was slightly loosened
        bottom_sum = np.sum(tested[25:, :] == 255)
        top_sum = np.sum(tested[:25, :] == 255)

        if top_sum != 0:
            ratio = bottom_sum/top_sum
        else:
            ratio = 1000

        min_value = min(sum_list)
        if ratio < 1.05 and ratio > 0.95:
            to_lower = [1,2,3,7,9,11,12,13,14,17,18,20,23,24,27,32,34]# for pattern letters ratio <1.1 and >0.9
            for lower in to_lower:
                sum_list[lower] = sum_list[lower]/2
        if ratio < 0.7:
            to_lower =[0,4,6,8,15,25,26,28,33]# for pattern letters ratio <0.85
            for lower in to_lower:
                sum_list[lower] = sum_list[lower]/2
        if ratio > 1.3:
            to_lower =[5,10,16,19,21,29,31]# for pattern letters ratio >1.15
            for lower in to_lower:
                sum_list[lower] = sum_list[lower]/2
        lowest_sum = 50000
        letter_found_index = 0

        for j, sum in enumerate(sum_list):
            if i > 1 or j > 9:
                if sum < lowest_sum:
                    letter_found_index = j
                    lowest_sum = sum
        if letter_found_index == 24 and i > 1:
            letter_found_index -= 15
        min_number = min(sum_list[:10])
        min_letter = min(sum_list[10:])


        #0 - Number #1 - Letter
        sum_lists.append(sum_list)

        if letter_found_index <= 9:
            real_chain = real_chain + '0'
        if letter_found_index > 9:
            real_chain = real_chain + '1'
        if letter_found_index <= 9 and lowest_sum + 200 < min_letter:
            chain = chain + '0'
        elif letter_found_index > 9 and lowest_sum + 200 < min_number:
            chain = chain + '1'
        else:
            chain = chain + 'x'


        output_string = output_string + letters_list[letter_found_index]

    #if there are less than 7 characters, return string now
    end_now = 0
    while len(output_string) < 7:
        end_now = 1
        output_string = output_string + '?'
    if end_now == 1:
        return output_string


    #7 - Try to find out if combination of letters and numbers can exist. If not, evaluate characters again by
    #increasing the scores of letters or numbers.
    if possible_plates_enabled == 1:
        # Possible combinations according to: https://pl.wikipedia.org/wiki/Tablice_rejestracyjne_w_Polsce
        # where 1 represents letter and 0 represents number (only last 5 are considered since first 2 are always a letter)
        chains = ['00000', '00001', '00011', '01000', '01100', '11000', '10011', '10100', '10010', '10110', '11100']
        Possible_chains = ['00000', '00001', '00011', '01000', '01100', '11000', '10011', '10100', '10010', '10110', '11100']

        #With enough data, this part helps to distinguish similar numbers and letters (8 and B, 5 and S, etc.)
        for Possible_chain in chains:
            for j in range(len(chain)-2):
                if chain[j+2] == 'x':
                    continue
                else:
                    if chain[j+2] != Possible_chain[j]:
                        Possible_chains.remove(Possible_chain)
                        break

        if len(Possible_chains) == 1:
            for i in range(len(Possible_chains[0])):
                if chain[i+2] == 'x':
                    to_modify = 0
                    if Possible_chains[0][i] == '0' and real_chain[i+2] == '1':
                        for j in range(len(sum_lists[i+2])):
                            if j > 9:
                                sum_lists[i+2][j] = sum_lists[i+2][j] + 2500
                                to_modify = 1

                    if Possible_chains[0][i] == '1' and real_chain[i+2] == '0':
                        for j in range(len(sum_lists[i+2])):
                            if j <= 9:
                                sum_lists[i+2][j] = sum_lists[i+2][j] + 2500
                                to_modify = 1
                    if to_modify == 1:
                        min_value = min(sum_lists[i + 2])
                        result = np.where(sum_lists[i + 2] == min_value)
                        s = list(output_string)
                        s[i+2] = letters_list[result[0][0]]
                        output_string = ''.join(s)




    if test_mode == 1:
        cv2.waitKey()
    cv2.destroyAllWindows()
    return output_string

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('images_dir', type=str)
    parser.add_argument('results_file', type=str)
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    results_file = Path(args.results_file)

    images_paths = sorted([image_path for image_path in images_dir.iterdir() if image_path.name.endswith('.jpg')])
    results = {}

    for image_path in images_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f'Error loading image {image_path}')
            continue
        results[image_path.name] = FindSomeLetters(image)





    with results_file.open('w') as output_file:
        json.dump(results, output_file, indent=4)


if __name__ == '__main__':
    main()