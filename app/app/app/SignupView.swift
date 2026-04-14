import SwiftUI

struct SignupView: View {
    @Binding var showSignup: Bool
    @Binding var showLogin: Bool
    
    @State private var name = ""
    @State private var email = ""
    @State private var password = ""
    
    var body: some View {
        VStack {
            Spacer()
            
            VStack(spacing: 20) {
                
                VStack(alignment: .leading) {
                    HStack {
                        Text("회원가입")
                            .font(.title)
                            .bold()
                            .foregroundColor(.white)
                        
                        Spacer()
                        
                        Button {
                            showSignup = false
                        } label: {
                            Image(systemName: "xmark.circle.fill")
                                .foregroundColor(.white)
                        }
                    }
                    
                    Text("새 계정을 만드세요")
                        .foregroundColor(.white.opacity(0.8))
                }
                .padding()
                .background(
                    LinearGradient(colors: [Color.blue, Color.purple],
                                   startPoint: .leading,
                                   endPoint: .trailing)
                )
                
                VStack(spacing: 15) {
                    CustomTextField(icon: "person", placeholder: "이름", text: $name)
                    CustomTextField(icon: "envelope", placeholder: "이메일", text: $email)
                    CustomSecureField(icon: "lock", placeholder: "비밀번호", text: $password)
                }
                .padding(.horizontal)
                
                Button("회원가입") {
                    signup()
                }
                .foregroundColor(.white)
                .frame(maxWidth: .infinity)
                .padding()
                .background(
                    LinearGradient(colors: [Color.blue, Color.purple],
                                   startPoint: .leading,
                                   endPoint: .trailing)
                )
                .cornerRadius(12)
                .padding(.horizontal)
                
                HStack {
                    Text("이미 계정이 있으신가요?")
                        .foregroundColor(.gray)
                    
                    Button("로그인") {
                        showSignup = false
                        showLogin = true
                    }
                    .foregroundColor(.blue)
                }
                .font(.footnote)
                
                Spacer()
            }
            .frame(height: 500)
            .background(Color.white)
            .cornerRadius(25)
        }
    }
    
    func signup() {
        print("회원가입 요청")
    }
}

struct SignupView_Previews: PreviewProvider {
    static var previews: some View {
        SignupView(
            showSignup: .constant(true),
            showLogin: .constant(false)
        )
    }
}
