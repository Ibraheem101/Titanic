FROM nginx:alpine

WORKDIR /app
COPY index.html .
COPY style.css .

EXPOSE 80  

COPY nginx.conf /etc/nginx/conf.d/default.conf

CMD ["nginx", "-g", "daemon off;"]