import os
import cv2
import torch
import lpips
import numpy as np
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as calculate_psnr
from skimage.metrics import structural_similarity as calculate_ssim
from pytorch_fid.fid_score import calculate_fid_given_paths


def evaluate_metrics(gt_dir, gen_dir, device):
    print("\n" + "="*40)
    print("Start evaluating metrics (PSNR, SSIM, LPIPS, FID)...")
    
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)
    
    image_names = [f for f in os.listdir(gt_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    image_names.sort()

    total_psnr, total_ssim, total_lpips = 0.0, 0.0, 0.0

    print(f"--> 1/2: Calculating PSNR, SSIM, LPIPS on {len(image_names)} images...")
    for img_name in tqdm(image_names):
        gt_path = os.path.join(gt_dir, img_name)
        gen_path = os.path.join(gen_dir, img_name)

        if not os.path.exists(gen_path):
            continue

        img_gt = cv2.imread(gt_path)
        img_gen = cv2.imread(gen_path)

        img_gt_rgb = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
        img_gen_rgb = cv2.cvtColor(img_gen, cv2.COLOR_BGR2RGB)

        total_psnr += calculate_psnr(img_gt_rgb, img_gen_rgb, data_range=255)
        total_ssim += calculate_ssim(img_gt_rgb, img_gen_rgb, channel_axis=2, data_range=255)

        tensor_gt = torch.from_numpy(img_gt_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device)
        tensor_gen = torch.from_numpy(img_gen_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device)
        tensor_gt = (tensor_gt / 255.0) * 2.0 - 1.0
        tensor_gen = (tensor_gen / 255.0) * 2.0 - 1.0

        with torch.no_grad():
            total_lpips += loss_fn_alex(tensor_gt, tensor_gen).item()

    num_imgs = len(image_names)
    avg_psnr = total_psnr / num_imgs
    avg_ssim = total_ssim / num_imgs
    avg_lpips = total_lpips / num_imgs

    print("--> 2/2: Calculating FID (this may take a while)...")
    fid_value = calculate_fid_given_paths([gt_dir, gen_dir], batch_size=16, device=device, dims=2048)

    print("\n" + "="*40)
    print("Final Evaluation Results:")
    print(f"Average PSNR  : {avg_psnr:.4f}")
    print(f"Average SSIM  : {avg_ssim:.4f}")
    print(f"Average LPIPS : {avg_lpips:.4f}")
    print(f"FID Score     : {fid_value:.4f}")
    print("="*40)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

 
    low_dir = './dataset/LOLv1/Test/Low'         
    gt_dir = './dataset/LOLv1/Test/High'        
    save_dir = './results/WaveLight/LOLv1'      
    os.makedirs(save_dir, exist_ok=True)

    
    # model = WaveLightNet().to(device)
    # model.load_state_dict(torch.load('./checkpoints/best_model.pth'))
    # model.eval()

    img_list = [f for f in os.listdir(low_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Start testing {len(img_list)} images...")
    
    with torch.no_grad():
        for img_name in tqdm(img_list, desc="Enhancing"):
            low_path = os.path.join(low_dir, img_name)
  
            # img_low = cv2.imread(low_path)
            # input_tensor = preprocess(img_low).to(device)
            # output_tensor = model(input_tensor)
            # img_enhanced = postprocess(output_tensor)
            
            save_path = os.path.join(save_dir, img_name)
            # cv2.imwrite(save_path, img_enhanced)
            pass 

    print("Inference finished. Images saved to:", save_dir)

  
    if os.path.exists(gt_dir) and len(os.listdir(gt_dir)) > 0:
        evaluate_metrics(gt_dir, save_dir, device)
    else:
        print("\nGround Truth folder not found or empty. Skipping PSNR/SSIM/LPIPS/FID evaluation.")
        print("For unpaired datasets (like DICM, LIME), please calculate NIQE and PI instead.")

if __name__ == '__main__':
    main()
