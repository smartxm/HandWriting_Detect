import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt


from PIL import Image, ImageDraw, ImageOps, ImageFont, ImageFilter, ImageTk
import tkinter as tk
from tkinter import Canvas, Button, Label, Scale, HORIZONTAL, Toplevel, messagebox, filedialog
import numpy as np
from tkinter import font as tkfont
import torch.nn.functional as F

class Net(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = torch.nn.Sequential(                   #(batch,1,28,28)
            torch.nn.Conv2d(1, 10, kernel_size=5),          #(batch,10,24,24)
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)               #(batch,10,12,12)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(10, 20, kernel_size=5),         #(batch,20,8,8)    
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),              #(batch,20,4,4)
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(320, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 10),
        )
        #self.dropout = self.Dropout(0.5)  随机杀死50%神经元,全连接层可以加

    def forward(self, x):

        batch_size = x.size(0)
        x = self.conv1(x)
        # print(f"conv1层结束的张量形状：{x.shape}")
        x = self.conv2(x)
        # print(f"conv2层结束的张量形状：{x.shape}")
        x = x.view(batch_size, -1)
        # print(f"view后的张量形状：{x.shape}")                          ## flatten 变成全连接网络需要的输入 (batch, 20,4,4) ==> (batch,320), -1 此处自动算出的是320
        x = self.fc(x)
        # print(f"Linear层结束的张量形状：{x.shape}")
        return x
    
def get_data_loader(is_train):
    to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])                     #将数据转换为pytorch的张量，像素值缩放，通道优先其中0.1307是mean均值和0.3081是std标准差
    data_set = MNIST("", is_train, transform=to_tensor, download=True)          #下载MNIST数据集
    return DataLoader(data_set, batch_size=15, shuffle=True)                    #shuffle：是否打乱，通常is_train为True时会进行打乱

def evaluate(test_data,net):
    n_correct = 0
    n_total = 0
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():                                   #禁用梯度计算，节省内存和计算资源（评估阶段不需要计算梯度）
        for(x, y) in test_data:                             #test_data：PyTorch的DataLoader对象 每次迭代返回：x：一个批次的图像数据，形状为 [batch_size, 1, 28, 28]y：对应的标签，形状为 [batch_size]
            # x, y = x.to(device), y.to(device)                           
            
            outputs = net.forward(x)        #重塑张量形状，将4D图像数据展平为2D（强制展平，不支持CNN） 原始：x 形状为 [batch_size, 1, 28, 28],展平后：[batch_size, 784] (因为 28×28 = 784),-1：自动计算该维度大小  然后执行前面定义的前向传播
            for i,output in enumerate(outputs):             #enumerate()：同时获取索引和值 i：当前样本在批次中的索引（0到batch_size-1） output：单个样本的预测结果，形状为 [10]
                if torch.argmax(output) == y[i]:            #遍历每个样本的预测，判断是否正确 torch.argmax(output)：找到张量中最大值的索引
                    n_correct += 1
                n_total += 1
    return n_correct / n_total


class HandwritingApp:
    def __init__(self, root, model):
        self.root = root
        self.model = model
        self.model.eval()  # Set model to evaluation mode
        
        self.root.title("Handwritten Digit Recognition")
        self.root.geometry("600x700")
        
        # Set background color
        self.root.configure(bg='light gray')
        
        # Title
        title_font = tkfont.Font(family="Helvetica", size=16, weight="bold")
        title_label = Label(root, text="Handwritten Digit Recognition System", 
                           font=title_font, bg='light gray')
        title_label.pack(pady=10)
        
        # Instructions
        instructions = Label(root, text="Draw a digit (0-9) in the canvas below, then click 'Recognize'", 
                            bg='light gray')
        instructions.pack(pady=5)
        
        # Canvas for drawing
        self.canvas_width = 280
        self.canvas_height = 280
        self.canvas = Canvas(root, width=self.canvas_width, height=self.canvas_height, 
                            bg='white', cursor="cross")
        self.canvas.pack(pady=10)
        
        # Bind mouse events
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonPress-1>", self.start_paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)
        
        # Brush size control
        brush_frame = tk.Frame(root, bg='light gray')
        brush_frame.pack(pady=5)
        
        brush_label = Label(brush_frame, text="Brush Size:", bg='light gray')
        brush_label.pack(side=tk.LEFT, padx=5)
        
        self.brush_size = Scale(brush_frame, from_=1, to=20, orient=HORIZONTAL, 
                               length=200, bg='light gray')
        self.brush_size.set(18)
        self.brush_size.pack(side=tk.LEFT)
        
        # Control buttons frame
        button_frame = tk.Frame(root, bg='light gray')
        button_frame.pack(pady=10)
        
        # Recognize button
        self.recognize_btn = Button(button_frame, text="Recognize", 
                                    command=self.recognize_digit, 
                                    bg='#4CAF50', fg='white',
                                    font=("Helvetica", 12), 
                                    width=10, height=2)
        self.recognize_btn.pack(side=tk.LEFT, padx=10)
        
        # Clear button
        self.clear_btn = Button(button_frame, text="Clear", 
                               command=self.clear_canvas,
                               bg='#f44336', fg='white',
                               font=("Helvetica", 12), 
                               width=10, height=2)
        self.clear_btn.pack(side=tk.LEFT, padx=10)
        
        # Save as MNIST button
        self.save_btn = Button(button_frame, text="Save as MNIST", 
                              command=self.save_as_mnist,
                              bg='#2196F3', fg='white',
                              font=("Helvetica", 12), 
                              width=10, height=2)
        self.save_btn.pack(side=tk.LEFT, padx=10)
        
        # Result display
        result_frame = tk.Frame(root, bg='light gray')
        result_frame.pack(pady=20)
        
        result_label = Label(result_frame, text="Prediction:", 
                            font=("Helvetica", 14, "bold"), 
                            bg='light gray')
        result_label.pack()
        
        self.result_text = Label(result_frame, text="", 
                                font=("Helvetica", 48, "bold"), 
                                fg='#FF5722', bg='light gray')
        self.result_text.pack()
        
        # Confidence display
        confidence_label = Label(result_frame, text="Confidence:", 
                                font=("Helvetica", 12), 
                                bg='light gray')
        confidence_label.pack()
        
        self.confidence_text = Label(result_frame, text="", 
                                    font=("Helvetica", 12), 
                                    fg='#3F51B5', bg='light gray')
        self.confidence_text.pack()
        
        # Status bar
        self.status_bar = Label(root, text="Draw a digit and click 'Recognize'", 
                               bd=1, relief=tk.SUNKEN, anchor=tk.W,
                               bg='light gray')
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Variables for drawing
        self.old_x = None
        self.old_y = None
        self.line_width = self.brush_size.get()
        
    def start_paint(self, event):
        self.old_x = event.x
        self.old_y = event.y
        
    def paint(self, event):
        self.line_width = self.brush_size.get()
        paint_color = 'black'
        
        if self.old_x and self.old_y:
            self.canvas.create_line(self.old_x, self.old_y, event.x, event.y,
                                   width=self.line_width, fill=paint_color,
                                   capstyle=tk.ROUND, smooth=tk.TRUE)
        self.old_x = event.x
        self.old_y = event.y
        
    def reset(self, event):
        self.old_x = None
        self.old_y = None
        
    def clear_canvas(self):
        self.canvas.delete("all")
        self.result_text.config(text="")
        self.confidence_text.config(text="")
        self.status_bar.config(text="Canvas cleared")
        
    def get_canvas_image(self):
        """Get image from canvas and process it to match MNIST format"""
        # Save canvas as PostScript and convert to PIL Image
        self.canvas.postscript(file="tmp_canvas.eps", colormode='color')
        img = Image.open("tmp_canvas.eps")
        
        # Convert to grayscale
        img = img.convert('L')
        
        # Resize to 28x28 (MNIST size)
        img = img.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Invert colors (MNIST has white digits on black background)
        img = ImageOps.invert(img)
        
        # Apply Gaussian blur to make it smoother (like MNIST)
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        # Increase contrast
        img = ImageOps.autocontrast(img)
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Normalize like MNIST (0-1 range with mean 0.1307 and std 0.3081)
        img_tensor = torch.from_numpy(img_array).float()
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        img_tensor = (img_tensor / 255.0 - 0.1307) / 0.3081
        
        return img, img_tensor
    
    def recognize_digit(self):
        try:
            # Get processed image
            pil_img, img_tensor = self.get_canvas_image()
            
            # Make prediction
            with torch.no_grad():
                output = self.model(img_tensor)
                probabilities = F.softmax(output, dim=1)
                predicted_digit = torch.argmax(output).item()
                confidence = probabilities[0][predicted_digit].item()
            
            # Update result display
            self.result_text.config(text=str(predicted_digit))
            self.confidence_text.config(text=f"{confidence*100:.2f}%")
            
            # Show top-3 predictions
            top3_probs, top3_indices = torch.topk(probabilities, 3)
            top3_info = "Top predictions: "
            for i in range(3):
                top3_info += f"{top3_indices[0][i].item()}({top3_probs[0][i].item()*100:.1f}%) "
            
            self.status_bar.config(text=f"Recognized as: {predicted_digit} with {confidence*100:.2f}% confidence. {top3_info}")
            
            # Show processed image in a new window
            self.show_processed_image(pil_img, predicted_digit, confidence)
            
        except Exception as e:
            messagebox.showerror("Error", f"Recognition failed: {str(e)}")
            self.status_bar.config(text="Recognition failed")
    
    def show_processed_image(self, pil_img, prediction, confidence):
        """Show the processed image in a new window"""
        new_window = Toplevel(self.root)
        new_window.title("Processed Image")
        new_window.geometry("400x300")
        
        # Convert PIL Image to PhotoImage
        img_display = pil_img.resize((140, 140), Image.Resampling.NEAREST)
        photo = ImageTk.PhotoImage(img_display)
        
        # Display image
        img_label = Label(new_window, image=photo)
        img_label.image = photo  # Keep a reference
        img_label.pack(pady=10)
        
        # Display prediction info
        info_label = Label(new_window, 
                          text=f"Prediction: {prediction}\nConfidence: {confidence*100:.2f}%",
                          font=("Helvetica", 12))
        info_label.pack(pady=10)
        
        # Note about processing
        note_label = Label(new_window, 
                          text="Image has been processed to match MNIST format:\n"
                               "1. Converted to grayscale\n"
                               "2. Resized to 28x28\n"
                               "3. Inverted colors\n"
                               "4. Normalized with MNIST statistics",
                          font=("Helvetica", 9),
                          justify=tk.LEFT)
        note_label.pack(pady=10)
    
    def save_as_mnist(self):
        """Save the drawn digit as MNIST-like image"""
        try:
            pil_img, _ = self.get_canvas_image()
            
            # Create save dialog
            from tkinter import filedialog
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
                initialfile="mnist_digit.png"
            )
            
            if file_path:
                # Save the image
                pil_img.save(file_path)
                
                # Also save the original drawing
                original_img = self.get_original_canvas_image()
                original_path = file_path.replace(".png", "_original.png")
                original_img.save(original_path)
                
                self.status_bar.config(text=f"Images saved: {file_path}")
                messagebox.showinfo("Success", 
                                  f"Images saved successfully!\n"
                                  f"MNIST-like: {file_path}\n"
                                  f"Original: {original_path}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save image: {str(e)}")
    
    def get_original_canvas_image(self):
        """Get the original canvas image without processing"""
        self.canvas.postscript(file="tmp_original.eps", colormode='color')
        img = Image.open("tmp_original.eps")
        return img.convert('RGB')


def main():

    isLoad = int(input("[0] 训练模型\n[1] 加载模型\n"))
    if isLoad == 0:
        train_data = get_data_loader(is_train=True)
        test_data = get_data_loader(is_train=False)
        net = Net()                                                         #创建模型
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # net = net.to(device)
        print("训练前正确率：", evaluate(test_data,net))
        optimizer = torch.optim.Adam(net.parameters(),lr=0.001)             #设置优化器（梯度下降算法）
        criterion = torch.nn.CrossEntropyLoss()
        for epoch in range(4):                                              #训练两轮
            for (x,y) in train_data:
                # x, y = x.to(device), y.to(device)
                net.zero_grad()                                             #清空梯度
                output = net.forward(x)
                loss = criterion(output, y)              
                optimizer.step()                                            #根据梯度和优化算法更新参数
            print("训练轮次：", epoch, "准确率", evaluate(test_data, net))

        print("Training completed!")
        isSave = input("是否保存模型([y]/n)?")
        if isSave == "y":
            torch.save(net.state_dict(), 'model.pth')
        # Create GUI application
        root = tk.Tk()
        app = HandwritingApp(root, net)
        root.mainloop()

    if isLoad == 1:
        net = Net()
        net.load_state_dict(torch.load('model.pth'))
        net.eval()
        root = tk.Tk()
        app = HandwritingApp(root, net)
        root.mainloop()

if __name__ == "__main__":
    main()
