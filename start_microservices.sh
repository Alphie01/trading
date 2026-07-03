#!/bin/bash
# -*- coding: utf-8 -*-
"""
Mikroservis Başlatma Script'i
AI Service ve Web Service'i sırayla başlatır
"""

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║                                                                    ║"
echo "║           🚀 CRYPTO TRADING MICROSERVICES LAUNCHER 🚀            ║"
echo "║                                                                    ║"
echo "╚════════════════════════════════════════════════════════════════════╝"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if directories exist
if [ ! -d "ai-service" ]; then
    echo -e "${RED}❌ ai-service directory not found${NC}"
    exit 1
fi

if [ ! -d "web-service" ]; then
    echo -e "${RED}❌ web-service directory not found${NC}"
    exit 1
fi

echo -e "${BLUE}📋 Mikroservis başlatma planı:${NC}"
echo "   1. AI Service (Port 8000)"
echo "   2. Web Service (Port 5000)"
echo ""

# Function to start AI Service
start_ai_service() {
    echo -e "${YELLOW}🤖 AI Service başlatılıyor...${NC}"
    cd ai-service
    
    # Check if virtual environment exists
    if [ ! -d "venv" ]; then
        echo -e "${YELLOW}📦 AI Service için virtual environment oluşturuluyor...${NC}"
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Install requirements if needed
    echo -e "${YELLOW}📦 AI Service dependencies kontrol ediliyor...${NC}"
    pip install -r requirements.txt > /dev/null 2>&1
    
    # Start AI Service in background
    echo -e "${GREEN}🚀 AI Service başlatılıyor (Background)...${NC}"
    nohup python run_ai_service.py > ai_service.log 2>&1 &
    AI_PID=$!
    
    echo "   AI Service PID: $AI_PID"
    echo "   Log: ai-service/ai_service.log"
    
    # Wait for AI Service to start
    echo -e "${YELLOW}⏳ AI Service'in başlaması bekleniyor...${NC}"
    sleep 10
    
    # Test AI Service
    if curl -s http://localhost:8000/health > /dev/null; then
        echo -e "${GREEN}✅ AI Service başarıyla başlatıldı (Port 8000)${NC}"
        cd ..
        return 0
    else
        echo -e "${RED}❌ AI Service başlatılamadı${NC}"
        kill $AI_PID 2>/dev/null
        cd ..
        return 1
    fi
}

# Function to start Web Service
start_web_service() {
    echo -e "${YELLOW}🌐 Web Service başlatılıyor...${NC}"
    cd web-service
    
    # Check if virtual environment exists
    if [ ! -d "venv" ]; then
        echo -e "${YELLOW}📦 Web Service için virtual environment oluşturuluyor...${NC}"
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Install requirements if needed
    echo -e "${YELLOW}📦 Web Service dependencies kontrol ediliyor...${NC}"
    pip install -r requirements.txt > /dev/null 2>&1
    
    # Start Web Service in background
    echo -e "${GREEN}🚀 Web Service başlatılıyor (Background)...${NC}"
    nohup python run_web_service.py > web_service.log 2>&1 &
    WEB_PID=$!
    
    echo "   Web Service PID: $WEB_PID"
    echo "   Log: web-service/web_service.log"
    
    # Wait for Web Service to start
    echo -e "${YELLOW}⏳ Web Service'in başlaması bekleniyor...${NC}"
    sleep 10
    
    # Test Web Service
    if curl -s http://localhost:5000 > /dev/null; then
        echo -e "${GREEN}✅ Web Service başarıyla başlatıldı (Port 5000)${NC}"
        cd ..
        return 0
    else
        echo -e "${RED}❌ Web Service başlatılamadı${NC}"
        kill $WEB_PID 2>/dev/null
        cd ..
        return 1
    fi
}

# Function to run tests
run_tests() {
    echo -e "${BLUE}🧪 Testler çalıştırılıyor...${NC}"
    
    # Test AI Service
    echo -e "${YELLOW}Testing AI Service...${NC}"
    python test_ai_service.py
    AI_TEST_RESULT=$?
    
    # Test Web Service
    echo -e "${YELLOW}Testing Web Service...${NC}"
    python test_web_service.py
    WEB_TEST_RESULT=$?
    
    if [ $AI_TEST_RESULT -eq 0 ] && [ $WEB_TEST_RESULT -eq 0 ]; then
        echo -e "${GREEN}✅ Tüm testler başarılı!${NC}"
        return 0
    else
        echo -e "${RED}❌ Bazı testler başarısız${NC}"
        return 1
    fi
}

# Main execution
main() {
    # Start AI Service
    if start_ai_service; then
        echo ""
        
        # Start Web Service
        if start_web_service; then
            echo ""
            echo -e "${GREEN}🎉 Mikroservisler başarıyla başlatıldı!${NC}"
            echo ""
            echo "📍 Service URLs:"
            echo "   AI Service:  http://localhost:8000"
            echo "   Web Service: http://localhost:5000"
            echo ""
            echo "📊 API Documentation:"
            echo "   FastAPI Docs: http://localhost:8000/docs"
            echo "   Web Dashboard: http://localhost:5000"
            echo ""
            
            # Ask if user wants to run tests
            read -p "🧪 Testleri çalıştırmak istiyor musunuz? (y/n): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                run_tests
            fi
            
            echo ""
            echo -e "${BLUE}💡 Servisleri durdurmak için:${NC}"
            echo "   pkill -f 'run_ai_service.py'"
            echo "   pkill -f 'run_web_service.py'"
            echo ""
            echo -e "${BLUE}📊 Logları izlemek için:${NC}"
            echo "   tail -f ai-service/ai_service.log"
            echo "   tail -f web-service/web_service.log"
            
        else
            echo -e "${RED}❌ Web Service başlatılamadı, AI Service durduruluyor...${NC}"
            pkill -f 'run_ai_service.py'
            exit 1
        fi
    else
        echo -e "${RED}❌ AI Service başlatılamadı${NC}"
        exit 1
    fi
}

# Handle Ctrl+C
trap 'echo -e "\n${YELLOW}🛑 Servisler durduruluyor...${NC}"; pkill -f "run_ai_service.py"; pkill -f "run_web_service.py"; exit 0' INT

# Run main function
main
