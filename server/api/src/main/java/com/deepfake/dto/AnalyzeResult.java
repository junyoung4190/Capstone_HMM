package com.deepfake.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.List;

public class AnalyzeResult {

    @JsonProperty("face_count")
    private int faceCount;

    private List<FaceDetail> faces;
    private Risk risk;

    public int getFaceCount() { return faceCount; }
    public List<FaceDetail> getFaces() { return faces; }
    public Risk getRisk() { return risk; }

    public static class FaceDetail {
        @JsonProperty("face_ratio")
        private double faceRatio;

        @JsonProperty("head_pose")
        private HeadPose headPose;

        public double getFaceRatio() { return faceRatio; }
        public HeadPose getHeadPose() { return headPose; }
    }

    public static class HeadPose {
        private double yaw;
        private double pitch;

        public double getYaw() { return yaw; }
        public double getPitch() { return pitch; }
    }

    public static class Risk {
        private double score;
        private String level;
        private String description; // ✅ 추가

        public double getScore() { return score; }
        public String getLevel() { return level; }
        public String getDescription() { return description; } // ✅ 추가
        public void setDescription(String description) { this.description = description; } // ✅ 추가
    }
}