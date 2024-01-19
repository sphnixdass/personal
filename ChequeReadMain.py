import MainPredection as MainPredection
import extractCharecters as extChar
import json
import os
#X:\\Coding\\Python\\ChequeRead\\Samples\\Temp\\20231117BOIInClearingBK1800HR-1.png
#r"X:\\Coding\\Python\\ChequeRead\\Samples\\Temp\\20231117BOI2InClearingBK1800HR-1.png"
#'X:\\Coding\\Python\\ChequeRead\\Samples\\Temp\\20231117BOIInClearingBK1800HR-3.png'

inputFolderPath = "X:\\Coding\\Python\\ChequeRead\\Samples\\Temp\\Testing\\"

all_files = os.listdir(inputFolderPath)
png_files = filter(lambda x: x[-4:] == ".png", all_files)
for png_file in png_files:
    try:
        inputTempFileNa = inputFolderPath + png_file
        print("Clearing old temp files....")
        extChar.delete_samples()
        print("Processing file name", inputTempFileNa)
        print("Extracting characters from image....")
        extChar.extract_char(inputTempFileNa)
        os.remove(inputTempFileNa)
        print("Artificial Intelligence deep learning neural networks model loading........")
        validation_dataset, prediction_model, num_to_char, images2 = MainPredection.load_mod()
        print("Artificial Intelligence predicting the characters........")
        resultcal = MainPredection.model_pred(validation_dataset, prediction_model,num_to_char, images2)

        # print("Dass", resultcal[0][2])

        filtered_list = list(filter(lambda row: int(row[2]) >= (int(resultcal[0][2]) - 150), resultcal))

        sorted_list = sorted(filtered_list, key=lambda x: int(x[1]))  # Sort by second column

        # print(sorted_list)

        outputList = []
        outputList2 = []
        precol = int(sorted_list[0][1])
        for x in sorted_list:

            print(x, x[1], precol, str(int(x[1]) - int(precol)))
            if float(x[4]) >= 98:
                if int(int(x[1]) - int(precol)) >=150 :
                    outputList.append("".join(outputList2))
                    outputList2.clear()
                    outputList2.append(x[0])
                else:
                    outputList2.append(x[0])
                precol = int(int(x[1]))

        outputList.append("".join(outputList2))
            # print(type(x))
        with open(inputTempFileNa + "_output.txt", "w") as f:
            f.write(json.dumps(outputList))
        print("*****************************************************************************")
        print("")
        print("Final output :",outputList, json.dumps(outputList))
        print("")
        print("*****************************************************************************")

        
        with open(inputTempFileNa + "_done.txt", "w") as f:
            f.write("success")
    except Exception as error:
        print("Exception occured ", inputTempFileNa, error)
        with open(inputTempFileNa + "_output.txt", "w") as f:
            f.write("error: unable to process the file")
        os.remove(inputTempFileNa)
        with open(inputTempFileNa + "_done.txt", "w") as f:
            f.write("success")