package com.deepfake.controller;

import com.deepfake.dto.AnalyzeResult;
import com.deepfake.dto.ImageResponse;
import com.deepfake.dto.ImageUploadResponse;
import com.deepfake.external.FaceShieldClient;
import com.deepfake.service.ImageService;

import jakarta.servlet.http.HttpServletRequest;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.List;

@RestController
@RequestMapping("/api/images")
public class ImageController {

    private final ImageService imageService;
    private final FaceShieldClient faceShieldClient;  

    @Autowired
    public ImageController(
            ImageService imageService,
            FaceShieldClient faceShieldClient         
    ) {
        this.imageService = imageService;
        this.faceShieldClient = faceShieldClient;     
    }

        @PostMapping("/analyze")
    public ResponseEntity<AnalyzeResult> analyze(
            @RequestParam("file") MultipartFile file
    ) {
        byte[] fileBytes;
        try {
            fileBytes = file.getBytes();
        } catch (IOException e) {
            throw new RuntimeException("파일 읽기 실패");
        }
        AnalyzeResult result = faceShieldClient.analyze(fileBytes, file.getOriginalFilename());
        return ResponseEntity.ok(result);
    }

    @PostMapping("/upload")
    public ResponseEntity<ImageUploadResponse> upload(
            HttpServletRequest request,
            @RequestParam("file") MultipartFile file
    ) {
        Long userId = (Long) request.getAttribute("userId");
        return ResponseEntity.ok(imageService.uploadImage(file, userId));
    }

    @GetMapping("/my")
    public ResponseEntity<List<ImageResponse>> getMyImages(HttpServletRequest request) {
        Long userId = (Long) request.getAttribute("userId");
        return ResponseEntity.ok(imageService.getMyImages(userId));
    }

    @PostMapping("/{imageId}/protect")
    public ResponseEntity<ImageResponse> protect(@PathVariable Long imageId) {
        return ResponseEntity.ok(imageService.protectImage(imageId));
    }

    @GetMapping("/{imageId}")
    public ResponseEntity<ImageResponse> getImageResult(@PathVariable Long imageId) {
        return ResponseEntity.ok(imageService.getImageResult(imageId));
    }
}