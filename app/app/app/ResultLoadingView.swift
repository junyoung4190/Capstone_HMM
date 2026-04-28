import SwiftUI

struct ResultLoadingView: View {
    let image: UIImage
    @Environment(\.dismiss) private var dismiss

    @State private var result: ImageResponse?
    @State private var isLoading = true
    @State private var isError = false

    var body: some View {
        Group {
            if let result = result {
                ResultView(result: result, onDismiss: { dismiss() })
                    .navigationBarHidden(true)
            } else {
                VStack(spacing: 20) {
                    if isLoading {
                        Text("AI 분석 중...")
                            .font(.title2)
                        ProgressView()
                    }
                    if isError {
                        Text("업로드 실패 😢")
                            .foregroundColor(.red)
                        Button("다시 시도") {
                            upload()
                        }
                    }
                }
            }
        }
        .navigationBarHidden(true)
        .onAppear { upload() }
    }

    func upload() {
        isLoading = true
        isError = false

        uploadImage(image: image) { res in
            isLoading = false
            if let res = res {
                print("✅ 성공: \(res)")
                self.result = res
            } else {
                print("❌ 실패: res가 nil")
                self.isError = true
            }
        }
    }
}
