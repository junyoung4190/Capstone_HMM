import SwiftUI

struct ResultLoadingView: View {
    let image: UIImage
    
    @State private var result: ImageResponse?
    @State private var goNext = false
    @State private var isLoading = true
    @State private var isError = false
    
    var body: some View {
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
            
            NavigationLink("", isActive: $goNext) {
                if let result = result {
                    ResultView(result: result)
                }
            }
        }
        .onAppear {
            upload()
        }
    }
    
    func upload() {
        isLoading = true
        isError = false
        
        uploadImage(image: image) { res in
            isLoading = false
            
            if let res = res {
                self.result = res
                self.goNext = true
            } else {
                self.isError = true
            }
        }
    }
}
