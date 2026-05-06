import 'package:flutter_test/flutter_test.dart';
import 'package:local_ai_chat/main.dart';

void main() {
  testWidgets('App smoke test', (WidgetTester tester) async {
    await tester.pumpWidget(const MyApp(onboarded: true));
  });
}