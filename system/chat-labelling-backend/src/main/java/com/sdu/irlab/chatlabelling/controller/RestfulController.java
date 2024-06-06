package com.sdu.irlab.chatlabelling.controller;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.sdu.irlab.chatlabelling.common.ConversationStatus;
import com.sdu.irlab.chatlabelling.datasource.domain.Conversation;
import com.sdu.irlab.chatlabelling.datasource.domain.Rate;
import com.sdu.irlab.chatlabelling.datasource.domain.Message;
import com.sdu.irlab.chatlabelling.datasource.domain.Recommend;
import com.sdu.irlab.chatlabelling.datasource.domain.User;
import com.sdu.irlab.chatlabelling.datasource.repository.ConversationDAO;
import com.sdu.irlab.chatlabelling.datasource.repository.UserDAO;
import com.sdu.irlab.chatlabelling.datasource.repository.RateDAO;
import com.sdu.irlab.chatlabelling.datasource.repository.MessageDAO;
import com.sdu.irlab.chatlabelling.datasource.repository.RecommendDAO;
import com.sdu.irlab.chatlabelling.security.CustomSessionManagement;
import com.sdu.irlab.chatlabelling.service.BaiduSearchService;
import com.sdu.irlab.chatlabelling.service.BingSearchService;
import com.sdu.irlab.chatlabelling.service.ChatService;
import com.sdu.irlab.chatlabelling.utils.ChatLabellingUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.web.authentication.WebAuthenticationDetails;
import org.springframework.security.web.context.HttpSessionSecurityContextRepository;
import org.springframework.web.bind.annotation.*;
import org.springframework.http.ResponseEntity;

import javax.annotation.PostConstruct;
import javax.servlet.http.HttpServletRequest;
import java.util.*;

import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;

@RestController
@RequestMapping("/api")
public class RestfulController {

    private ChatService chatService;
    private ConversationDAO conversationDAO;
    private UserDAO userDAO;
    private RateDAO rateDAO;
    private MessageDAO messageDAO;
    private RecommendDAO recommendDAO;

    private ObjectMapper objectMapper = new ObjectMapper();

    @Autowired
    protected AuthenticationManager authenticationManager;

//    @Autowired
//    private BingSearchService bingSearchService;

    @Autowired
    private BaiduSearchService baiduSearchService;

    @Value("${actionFile}")
    private String actionFile;
    private static String actionContent;

    @Value("${instructionFile}")
    private String instructionFile;
    private static String instructionContent;

    @Value("${searchResultConfigFile}")
    private String searchResultConfigFile;
    private static String searchResultConfigContent;

    @Autowired
    private CustomSessionManagement customSessionManagement;

//    @RequestMapping(value = "/currentUser", method = RequestMethod.GET)
//    public String currentUser(@RequestParam(value = "uuid") String uuid,
//                              @RequestParam(value = "user") String user) {
//        return user;
//    }
    /**
     * 这是初始化recommend_info的方法
     */
    @RequestMapping(value = "/loadRecommendInfo", method = RequestMethod.GET)
    public ResponseEntity<List<Map<String, String>>> loadRecommendInfo(
        @RequestParam("conversationId") Long conversationId,
        @RequestParam("user") String username) {

        Conversation conversation = conversationDAO.findById(conversationId).orElse(null);
        User user = userDAO.findByName(username);

        if (conversation == null || user == null) {
            // 处理参数无效的情况，可以返回适当的响应
            return ResponseEntity.badRequest().build();
        }

        List<Recommend> recommends = recommendDAO.findRecommendByConversationAndUserOrderByCreateTimeDesc(conversation, user);

        List<Map<String, String>> resultList = new ArrayList<>();

        for (Recommend recommend : recommends) {
            String evaluationObject = recommend.getEvaluationObject();  // 产品名
            String md5Value = recommend.getNameID();  // 产品md5编码

            Map<String, String> recommendation = new HashMap<>();
            recommendation.put("id", md5Value);
            recommendation.put("name", evaluationObject);

            resultList.add(recommendation);
        }

        return ResponseEntity.ok(resultList);
    }


    /*
    * 这是保存recommend_info的方法，防止网络问题导致推荐产品丢失
    */
    // 保存用户评分
    @RequestMapping(value = "/saveRecommend", method = RequestMethod.POST, headers = "Accept=application/json")
    public void saveRecommend(@RequestBody Map<String, Object> requestData) throws JsonProcessingException {
        System.out.println(requestData);
        List<Map<String, Object>> recommendInfoList = (List<Map<String, Object>>) requestData.get("recommend_info");
        String username = (String) requestData.get("username");
        Long conversationId = Long.valueOf((Integer) requestData.get("conversationId"));

        Conversation conversation = conversationDAO.getOne(conversationId);
        User user = userDAO.findByName(username);

        for (int i = 0; i < recommendInfoList.size(); i++) {
            Map<String, Object> recommendInfo = recommendInfoList.get(i);
            String evaluationObject = (String) recommendInfo.get("name");
            String nameID = (String) recommendInfo.get("id");

            Recommend recommend = new Recommend();
            recommend.setUser(user);
            recommend.setConversation(conversation);
            recommend.setEvaluationObject(evaluationObject);
            recommend.setNameID(nameID);

            if (recommendDAO != null) {
                recommendDAO.saveAndFlush(recommend);
            }
        }

    }

    @RequestMapping(value = "/instructions", method = RequestMethod.GET)
    public String instructions() {
        return instructionContent;
    }

    @RequestMapping(value = "/addLog", method = RequestMethod.POST, headers = "Accept=application/json")
    public void addLog(@RequestBody String req, @RequestParam(value = "_user") String user) {
        try {
            Map<String, Object> map = objectMapper.readValue(req, Map.class);
            chatService.createOperationLog(user, map.get("type").toString(), map.get("content").toString(), map.get("conversationId").toString());
        } catch (JsonProcessingException e) {
            e.printStackTrace();
        }
    }

    @RequestMapping(value = "/saveProfileAndLogin", method = RequestMethod.POST, headers = "Accept=application/json")
    public void saveProfileAndLogin(HttpServletRequest request, @RequestBody String req) {
        User u = new User();
        userDAO.saveAndFlush(u);//先保存一次，获取用户id
        String nameAndPass = "user-" + u.getId();
        u.setName(nameAndPass);
        u.setPassword(nameAndPass);
        u.setProfile(req);
        userDAO.saveAndFlush(u);
        UsernamePasswordAuthenticationToken token = new UsernamePasswordAuthenticationToken(u, nameAndPass, u.getAuthorities());
        token.setDetails(new WebAuthenticationDetails(request));
        Authentication authenticatedUser = authenticationManager.authenticate(token);
        SecurityContextHolder.getContext().setAuthentication(authenticatedUser);
        request.getSession().setAttribute(HttpSessionSecurityContextRepository.SPRING_SECURITY_CONTEXT_KEY, SecurityContextHolder.getContext());
    }

    /**
     * @throws JsonProcessingException
     */
    @RequestMapping(value = "/checkProfile", method = RequestMethod.GET)
    public boolean checkProfile(@RequestParam(value = "_user") String user) {
        User u = userDAO.findByName(user);
        return u.getProfile() == null || u.getProfile().length() == 0;
    }

    @RequestMapping(value = "/saveProfile", method = RequestMethod.POST, headers = "Accept=application/json")
    public void saveProfile(@RequestBody String req, @RequestParam(value = "_user") String user) {
        User u = userDAO.findByName(user);
        u.setProfile(req);
        userDAO.saveAndFlush(u);
    }

    // 保存用户评分
    @RequestMapping(value = "/saveRating", method = RequestMethod.POST, headers = "Accept=application/json")
    public void saveRating(@RequestBody Map<String, Object> requestData) throws JsonProcessingException {
        List<Map<String, Object>> recommendInfoList = (List<Map<String, Object>>) requestData.get("recommend_info");
        List<Integer> selectedOptions = (List<Integer>) requestData.get("selectedOptions");
        String role = (String) requestData.get("role");
        String userId = (String) requestData.get("userId");
        Long conversationId = Long.valueOf((Integer) requestData.get("conversationId"));

        Conversation conversation = conversationDAO.getOne(conversationId);
        User user = userDAO.findByName(userId);

        for (int i = 0; i < recommendInfoList.size(); i++) {
            Map<String, Object> recommendInfo = recommendInfoList.get(i);
            String evaluationObject = (String) recommendInfo.get("name");
            String nameID = (String) recommendInfo.get("id");

            Rate rate = new Rate();
            rate.setUser(user);
            rate.setConversation(conversation);
            rate.setEvaluationObject(evaluationObject);
            rate.setNameID(nameID);
            rate.setRelevance(selectedOptions.get(i));

            if (rateDAO != null) {
                rateDAO.saveAndFlush(rate);
            }
        }
    }

    // 保存系统评分
    @RequestMapping(value = "/saveRatingSys", method = RequestMethod.POST, headers = "Accept=application/json")
    public void saveRating(@RequestBody String req) throws JsonProcessingException {
        JsonNode node = objectMapper.readTree(req);
        String role = node.get("role").asText();
        long id = node.get("conversationId").asLong();
        Conversation conversation = conversationDAO.getOne(id);
        if (conversation == null) return;
        if (role.equals("sys")) {
            conversation.setSysRate(req);
        } else if (role.equals("cus")) {
            conversation.setCusRate(req);
        }
    }

    @RequestMapping(value = "/getSearchedMessage", method = RequestMethod.GET, headers = "Accept=application/json")
    public String getSearchedMessage(@RequestParam(value = "conversationId") Long id) {

        Conversation conversation = conversationDAO.getOne(id); // Create a Conversation object if needed
        System.out.println(conversation);
        String searchedMessage = "";
        // Use the provided conversationId and action to retrieve the message
        List<Message> messages = messageDAO.findMessageByConversationOrderByCreateTimeDesc(conversation);
//         Message message = messageDAO.findMessageByConversationAndActionOrderByCreateTimeDesc(conversation, "Recommend");
//         System.out.println(message);
        System.out.println(messages);
        String content = "";
        for (Message each_message : messages) {
            String action = each_message.getAction();
            System.out.println(action);
            if (action.equals("[\"Recommend\"]") || action.equals("[\"Recommend\",\"Factoid\"]") || action.equals("[\"Recommend\",\"Summary\"]") || action.equals("[\"Recommend\",\"Link\"]")) {
                content += each_message.getContent();
            }
        }
        if (content == "") {
            return null;
        }
        return content;

//         if (searchedMessage == null) {
//             searchedMessage = message.getContent();
//         }
//         String responseMessage = searchedMessage.isEmpty() ? "No message found." : searchedMessage;
//         return responseMessage;
    }

//    @RequestMapping(value = "/loadSearch", method = RequestMethod.POST, headers = "Accept=application/json", produces = "application/json;charset=UTF-8")
//    public String loadSearch(Principal principal, @RequestBody String req) {
//        try {
//            Map<String, Object> map = objectMapper.readValue(req, Map.class);
//            List<String> states = (List<String>) map.get("states");
//            String results = baiduSearchService.searchStates(states);
//            chatService.createSearchLog(principal.getName(), objectMapper.writeValueAsString(states), results, map.get("conversationId").toString());
//            return results;
//        } catch (JsonProcessingException e) {
//            e.printStackTrace();
//            return "";
//        }
//    }

    /**
     * 解析搜索结果，真正搜索过程放在了客户端，服务器只保存搜索后解析好的果
     *
     * @param req
     */
    @RequestMapping(value = "/saveSearchResults", method = RequestMethod.POST, headers = "Accept=application/json", produces = "application/json;charset=UTF-8")
    public void saveSearchResults(@RequestBody String req, @RequestParam(value = "_user") String user) {
        try {
            JsonNode node = objectMapper.readTree(req);
            chatService.createSearchLog(user, node.get("query").asText(), node.get("filter").asText(), node.get("data").toString(), node.get("conversationId").asText());
        } catch (JsonProcessingException e) {
            e.printStackTrace();
        }
    }


    @RequestMapping(value = "/loadCurrentState", method = RequestMethod.GET, produces = "application/json;charset=UTF-8")
    public String loadCurrentState(@RequestParam(value = "_user") String user) {
        Conversation conversation = chatService.getCurrentConversation(user);
        return conversation != null ? conversation.getState() : "[]";
    }

    @RequestMapping(value = "/loadSearchResultConfig", method = RequestMethod.GET, produces = "application/json;charset=UTF-8")
    public String loadSearchResultConfig() {
        return searchResultConfigContent;
    }

    @RequestMapping(value = "/loadActions", method = RequestMethod.GET, produces = "application/json;charset=UTF-8")
    public String loadActions() {
        return actionContent;
    }

    @RequestMapping(value = "/loadHistoryAndBackground", method = RequestMethod.GET, produces = "application/json;charset=UTF-8", headers = "Accept=application/json")
    public Map<String, Object> loadHistoryAndBackground(
            @RequestParam(name = "sysName") String sysName,
            @RequestParam(name = "cusName") String cusName,
            @RequestParam(name = "conversationId") Long conversationId) {


        int historyLimit = 5;
        Map<String, Object> map = new HashMap<>();
        Conversation conversation = conversationId != null ? conversationDAO.getOne(conversationId) : conversationDAO.findTopByCusUserAndSysUserAndStatusOrderByCreateTimeDesc(
                userDAO.findByName(cusName), userDAO.findByName(sysName), ConversationStatus.ONGOING);
        List<Message> history = chatService.loadHistory(conversation);
        if (history.size() > historyLimit) {
            history = history.subList(0, historyLimit);
        }
        map.put("history", history);
        map.put("background", conversation.getBackground());
        return map;
    }


    @PostConstruct
    private void readActions() {
        if (actionContent != null) return;
        String content = ChatLabellingUtils.readFileAsString(actionFile, "UTF-8");
        actionContent = content;

        if (instructionContent != null) return;
        content = ChatLabellingUtils.readFileAsString(instructionFile, "UTF-8");
        instructionContent = content;

        if (searchResultConfigContent != null) return;
        content = ChatLabellingUtils.readFileAsString(searchResultConfigFile, "UTF-8");
        searchResultConfigContent = content;
    }

    @Autowired
    public void setChatService(ChatService chatService) {
        this.chatService = chatService;
    }

    @Autowired
    public void setConversationDAO(ConversationDAO conversationDAO) {
        this.conversationDAO = conversationDAO;
    }

    @Autowired
    public void setUserDAO(UserDAO userDAO) {
        this.userDAO = userDAO;
    }

    @Autowired
    public void setRaterDAO(RateDAO rateDAO) {
        this.rateDAO = rateDAO;
    }

    @Autowired
    public void setMessageDAO(MessageDAO messageDAO) {
        this.messageDAO = messageDAO;
    }

    @Autowired
    public void setRecommendDAO(RecommendDAO recommendDAO) {
        this.recommendDAO = recommendDAO;
    }

}
